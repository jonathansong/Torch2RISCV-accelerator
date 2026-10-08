// ST engine: local memory -> DDR (docs/double_buffer_design.md §5).
//
// Same shapes and burst splitting as sa_ld (LINEAR only). Bursts are dealt
// round-robin to NPORTS AXI write ports; a burst's local start position is
// queued for its port once its AW handshake is done, so W data always
// follows its AW. One local read per cycle (1-cycle latency) feeds small
// per-port skid FIFOs that drive the W channels. A command completes when
// all of its B responses are back; `err` reports any SLVERR/DECERR.
//
// DMA_W = 128 (see sa_ld): bursts start at the address rounded down to 16
// bytes; WSTRB leaves out the lower lane of a first beat or the upper lane of
// a last beat that are not the command's. A beat's two lanes come from one
// local read when they fall into one word at an even lane, else from two
// consecutive reads of the same port.
`timescale 1ns / 1ps
`include "sa_macros.vh"

module sa_st #(
    parameter integer D      = 8,
    parameter integer NPORTS = 1,
    parameter integer DMA_W  = 64                    // AXI data width: 64 or 128
) (
    input  wire                 clk,
    input  wire                 resetn,

    input  wire                 cmd_valid,
    output wire                 cmd_ready,
    input  wire [31:0]          cmd_ddr,
    input  wire [3:0]           cmd_mem,
    input  wire [15:0]          cmd_word,
    input  wire [15:0]          cmd_rows,
    input  wire [15:0]          cmd_row_bytes,
    input  wire [31:0]          cmd_pitch,
    output reg                  done,
    output reg                  err,
    output wire                 busy,

    // local read (data one cycle later, full word of the addressed memory)
    output wire                 lr_en,
    output reg  [3:0]           lr_mem,
    output wire [15:0]          lr_word,
    input  wire [8*D-1:0]       lr_spad_a,
    input  wire [8*D-1:0]       lr_spad_b,
    input  wire [32*D-1:0]      lr_acc,

    output reg  [NPORTS*32-1:0] m_awaddr,
    output reg  [NPORTS*8-1:0]  m_awlen,
    output reg  [NPORTS-1:0]    m_awvalid,
    input  wire [NPORTS-1:0]    m_awready,
    output wire [NPORTS*DMA_W-1:0]   m_wdata,
    output wire [NPORTS*DMA_W/8-1:0] m_wstrb,
    output wire [NPORTS-1:0]    m_wlast,
    output wire [NPORTS-1:0]    m_wvalid,
    input  wire [NPORTS-1:0]    m_wready,
    input  wire [NPORTS*2-1:0]  m_bresp,
    input  wire [NPORTS-1:0]    m_bvalid,
    output wire [NPORTS-1:0]    m_bready,

    // performance events: {W stalled by the port, beat accepted, command active};
    // perf_two: the accepted beat carries two lanes (counts twice: 8-byte units)
    output wire [2:0]           perf_ev,
    output wire                 perf_two
);
    `include "sa_defs.vh"
    localparam integer LOGD = $clog2(D);
    localparam integer PB   = NPORTS > 1 ? $clog2(NPORTS) : 1;
    localparam integer QD   = 8;          // queued bursts per port
    localparam integer SD   = 4;          // skid entries per port
    localparam integer BL   = DMA_W / 64; // lanes per beat (1 or 2)
    localparam integer W2   = BL == 2;

    function [3:0] lpw_log2(input [3:0] mem);
        lpw_log2 = mem == MEM_ACC ? LOGD - 1 : LOGD - 3;
    endfunction

    // ---------------------------------------------------------- command
    reg         active, gen;
    reg  [15:0] rows;
    reg  [31:0] row_bytes;       // may be flattened: rows x row_bytes
    reg  [31:0] pitch;
    reg  [3:0]  lpwl;
    reg  [15:0] wpr;
    reg  [15:0] r;
    reg  [31:0] off;
    reg  [31:0] row_ddr;
    reg  [15:0] row_word, cur_word;
    reg  [7:0]  cur_lane;
    reg  [31:0] issued, completed;
    reg         berr;

    assign cmd_ready = !active && resetn;

    wire [3:0] c_lpwl = lpw_log2(cmd_mem);
    wire       flat   = cmd_pitch == cmd_row_bytes &&
                        (cmd_row_bytes & ((16'd8 << c_lpwl) - 16'd1)) == 0;
    assign busy      = active;

    // next burst, in lanes (as sa_ld)
    wire [31:0] addr      = row_ddr + off;
    wire        s0        = W2 ? addr[3] : 1'b0;
    wire [31:0] lanes_row = (row_bytes - off) >> 3;
    wire [12:0] lanes_4k  = (13'h1000 - {1'b0, addr[11:0]}) >> 3;
    wire [5:0]  lanes_max = 6'd16 * BL - s0;
    wire [5:0]  nl        = lanes_row < lanes_max && lanes_row <= lanes_4k ? lanes_row[5:0] :
                            lanes_4k < lanes_max ? lanes_4k[5:0] : lanes_max;
    wire [6:0]  nbeat     = W2 ? (s0 + nl + 7'd1) >> 1 : nl;
    wire        tfull     = W2 ? !(s0 ^ nl[0]) : 1'b1;
    wire [8:0]  lane_sum  = cur_lane + nl;
    wire [7:0]  lane_mask = (8'd1 << lpwl) - 8'd1;

    // burst waiting for its AW handshake (per port), then queued for W
    reg  [15:0] aw_word [0:NPORTS-1];
    reg  [7:0]  aw_lane [0:NPORTS-1];
    reg  [4:0]  aw_beats[0:NPORTS-1];
    reg         aw_s0   [0:NPORTS-1];
    reg         aw_tf   [0:NPORTS-1];
    reg  [15:0] q_word  [0:NPORTS-1][0:QD-1];
    reg  [7:0]  q_lane  [0:NPORTS-1][0:QD-1];
    reg  [4:0]  q_beats [0:NPORTS-1][0:QD-1];
    reg         q_s0    [0:NPORTS-1][0:QD-1];
    reg         q_tf    [0:NPORTS-1][0:QD-1];
    reg  [3:0]  q_wp    [0:NPORTS-1];
    reg  [3:0]  q_rp    [0:NPORTS-1];
    reg  [PB-1:0] aw_rr;

    wire [NPORTS-1:0] q_full, q_empty;
    genvar gp;
    generate
        for (gp = 0; gp < NPORTS; gp = gp + 1) begin : qst
            // 4-bit difference: modulo-16 pointers (a 32-bit compare breaks on wrap);
            // an AW in flight also owns a queue slot
            wire [3:0] used = q_wp[gp] - q_rp[gp];
            assign q_full[gp]  = used + m_awvalid[gp] >= QD;
            assign q_empty[gp] = used == 0;
        end
    endgenerate

    wire issue_ok = gen && !m_awvalid[aw_rr] && !q_full[aw_rr];

    // ------------------------------------------------ W side declarations
    reg  [PB-1:0] w_rr, w_sel;
    reg           w_any;
    reg  [4:0]    beat_idx [0:NPORTS-1];
    reg  [2:0]    sk_cnt   [0:NPORTS-1];      // entries + beats being read
    reg  [DMA_W-1:0]   sk_data [0:NPORTS-1][0:SD-1];
    reg  [DMA_W/8-1:0] sk_strb [0:NPORTS-1][0:SD-1];
    reg           sk_last  [0:NPORTS-1][0:SD-1];
    reg  [2:0]    sk_wp    [0:NPORTS-1];
    reg  [2:0]    sk_rp    [0:NPORTS-1];
    reg           w_hold;                     // the selected port's beat: its upper lane next
    reg  [PB-1:0] w_hport;

    integer p;
    always @(posedge clk) begin
        done <= 0;
        if (!resetn) begin
            active    <= 0;
            gen       <= 0;
            m_awvalid <= 0;
            aw_rr     <= 0;
            for (p = 0; p < NPORTS; p = p + 1) q_wp[p] <= 0;
        end else begin
            for (p = 0; p < NPORTS; p = p + 1)
                if (m_awvalid[p] && m_awready[p]) begin
                    m_awvalid[p]                 <= 0;
                    q_word[p][q_wp[p][2:0]]      <= aw_word[p];
                    q_lane[p][q_wp[p][2:0]]      <= aw_lane[p];
                    q_beats[p][q_wp[p][2:0]]     <= aw_beats[p];
                    q_s0[p][q_wp[p][2:0]]        <= aw_s0[p];
                    q_tf[p][q_wp[p][2:0]]        <= aw_tf[p];
                    q_wp[p]                      <= q_wp[p] + 1;
                end

            if (cmd_valid && cmd_ready) begin
                active    <= 1;
                gen       <= 1;
                // contiguous rows that fill whole words: one long row (see sa_ld)
                if (flat) begin
                    rows      <= 1;
                    row_bytes <= cmd_rows * cmd_row_bytes;
                end else begin
                    rows      <= cmd_rows;
                    row_bytes <= cmd_row_bytes;
                end
                pitch     <= cmd_pitch;
                lr_mem    <= cmd_mem;
                lpwl      <= lpw_log2(cmd_mem);
                wpr       <= (cmd_row_bytes + (16'd8 << lpw_log2(cmd_mem)) - 16'd1) >> (lpw_log2(cmd_mem) + 3);
                r         <= 0;
                off       <= 0;
                row_ddr   <= cmd_ddr;
                row_word  <= cmd_word;
                cur_word  <= cmd_word;
                cur_lane  <= 0;
                issued    <= 0;
                berr      <= 0;
            end else if (issue_ok) begin
                m_awvalid[aw_rr]          <= 1;
                m_awaddr[32*aw_rr +: 32]  <= W2 ? {addr[31:4], 4'd0} : addr;
                m_awlen[8*aw_rr +: 8]     <= nbeat - 1;
                aw_word[aw_rr]            <= cur_word;
                aw_lane[aw_rr]            <= cur_lane;
                aw_beats[aw_rr]           <= nbeat[4:0];
                aw_s0[aw_rr]              <= s0;
                aw_tf[aw_rr]              <= tfull;
                issued                    <= issued + 1;
                aw_rr <= aw_rr == NPORTS - 1 ? 0 : aw_rr + 1;
                if (off + {nl, 3'b000} == row_bytes) begin
                    off      <= 0;
                    r        <= r + 1;
                    row_ddr  <= row_ddr + pitch;
                    row_word <= row_word + wpr;
                    cur_word <= row_word + wpr;
                    cur_lane <= 0;
                    if (r + 1 == rows) gen <= 0;
                end else begin
                    off      <= off + {nl, 3'b000};
                    cur_word <= cur_word + (lane_sum >> lpwl);
                    cur_lane <= lane_sum[7:0] & lane_mask;
                end
            end else if (NPORTS > 1 && gen && (m_awvalid[aw_rr] || q_full[aw_rr])) begin
                aw_rr <= aw_rr == NPORTS - 1 ? 0 : aw_rr + 1;
            end

            if (active && !gen && completed == issued && !(cmd_valid && cmd_ready)) begin
                active <= 0;
                done   <= 1;
                err    <= berr;
            end
            for (p = 0; p < NPORTS; p = p + 1)
                if (m_bvalid[p] && m_bresp[2*p +: 2] != 2'b00) berr <= 1;
        end
    end

    // ----------------------------------------------- local read -> skid
    // a beat starts when its port's skid has room (counted from its first
    // read); a beat read in two cycles keeps its port (w_hold)
    integer s, cand;
    always @* begin
        w_any = 0;
        w_sel = 0;
        if (w_hold) begin
            w_any = 1;
            w_sel = w_hport;
        end else
            for (s = NPORTS - 1; s >= 0; s = s - 1) begin
                cand = (w_rr + s) % NPORTS;
                if (!q_empty[cand] && sk_cnt[cand] < SD) begin
                    w_any = 1;
                    w_sel = cand;
                end
            end
    end

    wire [15:0] h_word  = q_word[w_sel][q_rp[w_sel][2:0]];
    wire [7:0]  h_lane  = q_lane[w_sel][q_rp[w_sel][2:0]];
    wire [4:0]  h_beats = q_beats[w_sel][q_rp[w_sel][2:0]];
    wire        h_s0    = q_s0[w_sel][q_rp[w_sel][2:0]];
    wire        h_tf    = q_tf[w_sel][q_rp[w_sel][2:0]];
    wire        h_last  = beat_idx[w_sel] == h_beats - 1;
    wire        v0      = !(beat_idx[w_sel] == 0 && h_s0);
    wire        v1      = W2 && !(h_last && !h_tf);
    wire [8:0]  h_sum0  = h_lane + ({4'd0, beat_idx[w_sel]} << W2) - h_s0;
    wire        two     = v0 && v1;
    wire        merge   = two && lpwl != 0 && !h_sum0[0];
    wire        upper   = w_hold || !v0;
    wire [8:0]  r_sum   = h_sum0 + upper;
    wire        w_done  = !(two && !merge && !w_hold);    // this read completes the beat

    assign lr_en   = w_any;
    assign lr_word = h_word + (r_sum >> lpwl);

    // B responses this cycle (several ports can answer together)
    reg [3:0] b_count;
    integer bc;
    always @* begin
        b_count = 0;
        for (bc = 0; bc < NPORTS; bc = bc + 1) b_count = b_count + m_bvalid[bc];
    end

    // read in flight: which port / lane / beat position
    reg          rd_v;
    reg [PB-1:0] rd_port;
    reg [7:0]    rd_lane;
    reg          rd_last, rd_merge, rd_upper, rd_done;
    reg [1:0]    rd_strb;               // the beat's lanes that are the command's
    reg [63:0]   asm_lo;                // a two-read beat: its lower lane
    wire [63:0]  rd_lo = lr_mem == MEM_ACC ? lr_acc[64*rd_lane +: 64] :
                         lr_mem == MEM_SPAD_B ? lr_spad_b[64*rd_lane +: 64] :
                                                lr_spad_a[64*rd_lane +: 64];
    wire [7:0]   rd_lane1 = (rd_lane + 8'd1) & lane_mask;
    wire [63:0]  rd_hi = lr_mem == MEM_ACC ? lr_acc[64*rd_lane1 +: 64] :
                         lr_mem == MEM_SPAD_B ? lr_spad_b[64*rd_lane1 +: 64] :
                                                lr_spad_a[64*rd_lane1 +: 64];
    wire [DMA_W-1:0] rd_beat;
    generate
        if (W2) begin : beat2
            assign rd_beat = rd_merge ? {rd_hi, rd_lo} : rd_upper ? {rd_lo, asm_lo} : {rd_lo, rd_lo};
        end else begin : beat1
            assign rd_beat = rd_lo;
        end
    endgenerate

    generate
        for (gp = 0; gp < NPORTS; gp = gp + 1) begin : wch
            assign m_wvalid[gp]                 = sk_wp[gp] != sk_rp[gp];
            assign m_wdata[DMA_W*gp +: DMA_W]   = sk_data[gp][sk_rp[gp][1:0]];
            assign m_wstrb[DMA_W/8*gp +: DMA_W/8] = sk_strb[gp][sk_rp[gp][1:0]];
            assign m_wlast[gp]                  = sk_last[gp][sk_rp[gp][1:0]];
            assign m_bready[gp]                 = 1'b1;
        end
    endgenerate

    always @(posedge clk) begin
        if (!resetn) begin
            w_rr      <= 0;
            w_hold    <= 0;
            rd_v      <= 0;
            completed <= 0;
            for (p = 0; p < NPORTS; p = p + 1) begin
                q_rp[p] <= 0; beat_idx[p] <= 0; sk_cnt[p] <= 0; sk_wp[p] <= 0; sk_rp[p] <= 0;
            end
        end else begin
            if (cmd_valid && cmd_ready) completed <= 0;
            else                        completed <= completed + b_count;

            // issue a local read
            rd_v <= w_any;
            if (w_any) begin
                rd_port  <= w_sel;
                rd_lane  <= r_sum[7:0] & lane_mask;
                rd_last  <= h_last;
                rd_merge <= merge;
                rd_upper <= upper;
                rd_done  <= w_done;
                rd_strb  <= {v1, v0};
                w_hold   <= !w_done;
                w_hport  <= w_sel;
                if (w_done) begin
                    w_rr <= w_sel == NPORTS - 1 ? 0 : w_sel + 1;
                    if (h_last) begin
                        beat_idx[w_sel] <= 0;
                        q_rp[w_sel]     <= q_rp[w_sel] + 1;
                    end else
                        beat_idx[w_sel] <= beat_idx[w_sel] + 1;
                end
            end
            // data returns: complete beats go into the skid of their port
            if (rd_v && !rd_done) asm_lo <= rd_lo;
            if (rd_v && rd_done) begin
                sk_data[rd_port][sk_wp[rd_port][1:0]] <= rd_beat;
                sk_strb[rd_port][sk_wp[rd_port][1:0]] <= W2 ? {{8{rd_strb[1]}}, {8{rd_strb[0]}}} : {DMA_W/8{1'b1}};
                sk_last[rd_port][sk_wp[rd_port][1:0]] <= rd_last;
                sk_wp[rd_port] <= sk_wp[rd_port] + 1;
            end
            // occupancy: +1 at a beat's first read, -1 when its W beat leaves
            for (p = 0; p < NPORTS; p = p + 1) begin
                sk_cnt[p] <= sk_cnt[p] + (w_any && !w_hold && w_sel == p) - (m_wvalid[p] && m_wready[p]);
                if (m_wvalid[p] && m_wready[p]) sk_rp[p] <= sk_rp[p] + 1;
            end
        end
    end

    assign perf_ev = {|(m_wvalid & ~m_wready), |(m_wvalid & m_wready), active};
    // the accepted beat's strobes cover both lanes (one port at a time accepts: NPORTS = 1 exact)
    reg two_acc;
    integer t;
    always @* begin
        two_acc = 0;
        for (t = 0; t < NPORTS; t = t + 1)
            if (m_wvalid[t] && m_wready[t] && W2 && m_wstrb[DMA_W/8*t] && m_wstrb[DMA_W/8*t + 8]) two_acc = 1;
    end
    assign perf_two = two_acc;
endmodule
