// LD engine: DDR -> local memory (docs/double_buffer_design.md §5).
//
// A command moves `rows` DDR rows of `row_bytes` (multiple of 8) spaced by
// `pitch`, starting at `ddr`, into memory `mem` at word `word`:
//   LINEAR     each DDR row starts at a new local word; bytes fill words in
//              order (lane = 64-bit piece of a word)
//   INTERLEAVE local word = word + chunk*rows + row, chunk = word-sized piece
//              of a row (A strips for the array; SPAD only)
// It is split into INCR bursts of <= 16 beats that never cross 4 KB, dealt
// round-robin to NPORTS AXI read ports (up to QD = 8 outstanding per port).
// Each burst remembers its local start position, so returning beats are
// written by address in any port order. Commands run one at a time; `done`
// pulses after the last beat of the command is written, with `err` set if any
// beat came back with SLVERR/DECERR.
//
// DMA_W = 128 (docs/kv260_upgrade_plan.md K2a): a beat holds two lanes. The
// addresses and lengths keep their 8-byte granularity: a burst starts at its
// address rounded down to 16 bytes, so its first beat may hold only its upper
// lane and its last beat only its lower one (the burst queue keeps both
// facts). A beat whose two lanes fall into one local word at an even lane is
// written in one cycle (lw_two; e.g. SPAD at D = 16 from a 16-byte aligned
// address); otherwise its lanes are written one per cycle (R held one cycle).
`timescale 1ns / 1ps
`include "sa_macros.vh"

module sa_ld #(
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
    input  wire [1:0]           cmd_mode,
    output reg                  done,
    output reg                  err,
    output wire                 busy,

    // local write: one 64-bit lane of a word per cycle, or two (lw_two: lanes
    // lw_lane, lw_lane + 1 of one word, lw_lane even, data {lane + 1, lane});
    // one lane: its data in every 64-bit piece of lw_data
    output wire                 lw_en,
    output reg  [3:0]           lw_mem,
    output wire [15:0]          lw_word,
    output wire [7:0]           lw_lane,
    output wire                 lw_two,
    output wire [DMA_W-1:0]     lw_data,

    output reg  [NPORTS*32-1:0] m_araddr,
    output reg  [NPORTS*8-1:0]  m_arlen,
    output reg  [NPORTS-1:0]    m_arvalid,
    input  wire [NPORTS-1:0]    m_arready,
    input  wire [NPORTS*DMA_W-1:0] m_rdata,
    input  wire [NPORTS*2-1:0]  m_rresp,
    input  wire [NPORTS-1:0]    m_rlast,
    input  wire [NPORTS-1:0]    m_rvalid,
    output wire [NPORTS-1:0]    m_rready,

    // performance events: {AR stalled by the port, local write, command active};
    // perf_two: the local write is two lanes (counts twice: 8-byte units)
    output wire [2:0]           perf_ev,
    output wire                 perf_two
);
    `include "sa_defs.vh"
    localparam integer LOGD = $clog2(D);
    localparam integer PB   = NPORTS > 1 ? $clog2(NPORTS) : 1;
    localparam integer QD   = 8;                      // outstanding bursts per port
    localparam integer BL   = DMA_W / 64;             // lanes per beat (1 or 2)
    localparam integer W2   = BL == 2;

    // lanes per word (log2) of a memory
    function [3:0] lpw_log2(input [3:0] mem);
        lpw_log2 = mem == MEM_ACC  ? LOGD - 1 :       // 4D bytes = D/2 lanes
                   mem == MEM_DESC ? 0 : LOGD - 3;    // SPAD: D bytes = D/8 lanes
    endfunction

    // ---------------------------------------------------------- command
    reg         active;           // command accepted, not yet done
    reg         gen;              // still issuing bursts
    reg  [15:0] rows;
    reg  [31:0] row_bytes;       // may be flattened: rows x row_bytes
    reg  [31:0] pitch;
    reg  [3:0]  lpwl;
    reg  [15:0] step;             // local words between consecutive chunks of a row
    reg  [15:0] wpr;              // LINEAR: words per row
    reg  [15:0] r;                // current row
    reg  [31:0] off;              // byte offset in the row
    reg  [31:0] row_ddr;
    reg  [15:0] row_word;
    reg  [15:0] cur_word;
    reg  [7:0]  cur_lane;
    reg  [31:0] issued, completed;
    reg         rerr;

    assign cmd_ready = !active && resetn;

    wire [3:0] c_lpwl = lpw_log2(cmd_mem);
    wire       flat   = !cmd_mode[MODE_INTERLEAVE] && cmd_pitch == cmd_row_bytes &&
                        (cmd_row_bytes & ((16'd8 << c_lpwl) - 16'd1)) == 0;
    assign busy      = active;

    // next burst, in lanes: <= 16 beats, not past the row or the 4 KB boundary
    wire [31:0] addr      = row_ddr + off;
    wire        s0        = W2 ? addr[3] : 1'b0;        // first beat: lower lane not ours
    wire [31:0] lanes_row = (row_bytes - off) >> 3;
    wire [12:0] lanes_4k  = (13'h1000 - {1'b0, addr[11:0]}) >> 3;
    wire [5:0]  lanes_max = 6'd16 * BL - s0;
    wire [5:0]  nl        = lanes_row < lanes_max && lanes_row <= lanes_4k ? lanes_row[5:0] :
                            lanes_4k < lanes_max ? lanes_4k[5:0] : lanes_max;
    wire [6:0]  nbeat     = W2 ? (s0 + nl + 7'd1) >> 1 : nl;
    wire        tfull     = W2 ? !(s0 ^ nl[0]) : 1'b1;  // last beat: upper lane ours too
    wire [8:0]  lane_sum  = cur_lane + nl;
    wire [15:0] words_adv = lane_sum >> lpwl;
    wire [7:0]  lane_mask = (8'd1 << lpwl) - 8'd1;

    // per-port burst queues: {word, lane, first-beat skip, last beat full}
    reg  [15:0] q_word  [0:NPORTS-1][0:QD-1];
    reg  [7:0]  q_lane  [0:NPORTS-1][0:QD-1];
    reg         q_s0    [0:NPORTS-1][0:QD-1];
    reg         q_tf    [0:NPORTS-1][0:QD-1];
    reg  [3:0]  q_wp    [0:NPORTS-1];
    reg  [3:0]  q_rp    [0:NPORTS-1];
    wire [NPORTS-1:0] q_full, q_empty;

    reg  [PB-1:0] ar_rr;          // next port to issue on
    reg  [PB-1:0] r_rr;           // R side: round-robin start
    reg  [PB-1:0] r_sel;          // R side: port granted this cycle
    reg           r_any;
    wire          r_take;         // R side: the granted beat is consumed this cycle

    genvar gp;
    generate
        for (gp = 0; gp < NPORTS; gp = gp + 1) begin : qst
            // 4-bit difference: modulo-16 pointers (a 32-bit compare breaks on wrap)
            wire [3:0] used = q_wp[gp] - q_rp[gp];
            assign q_full[gp]  = used == QD;
            assign q_empty[gp] = used == 0;
        end
    endgenerate

    wire issue_ok = gen && !m_arvalid[ar_rr] && !q_full[ar_rr];

    integer p;
    always @(posedge clk) begin
        done <= 0;
        if (!resetn) begin
            active    <= 0;
            gen       <= 0;
            m_arvalid <= 0;
            ar_rr     <= 0;
            for (p = 0; p < NPORTS; p = p + 1) q_wp[p] <= 0;
        end else begin
            for (p = 0; p < NPORTS; p = p + 1)
                if (m_arvalid[p] && m_arready[p]) m_arvalid[p] <= 0;

            if (cmd_valid && cmd_ready) begin
                active    <= 1;
                gen       <= 1;
                // contiguous rows that fill whole words map identically as one
                // long row: fewer, longer bursts (e.g. an 8x8 tile = 1 burst)
                if (flat) begin
                    rows      <= 1;
                    row_bytes <= cmd_rows * cmd_row_bytes;
                end else begin
                    rows      <= cmd_rows;
                    row_bytes <= cmd_row_bytes;
                end
                pitch     <= cmd_pitch;
                lw_mem    <= cmd_mem;
                lpwl      <= lpw_log2(cmd_mem);
                step      <= cmd_mode[MODE_INTERLEAVE] ? cmd_rows : 16'd1;
                wpr       <= (cmd_row_bytes + (16'd8 << lpw_log2(cmd_mem)) - 16'd1) >> (lpw_log2(cmd_mem) + 3);
                r         <= 0;
                off       <= 0;
                row_ddr   <= cmd_ddr;
                row_word  <= cmd_word;
                cur_word  <= cmd_word;
                cur_lane  <= 0;
                issued    <= 0;
                rerr      <= 0;
            end else if (issue_ok) begin
                m_arvalid[ar_rr]            <= 1;
                m_araddr[32*ar_rr +: 32]    <= W2 ? {addr[31:4], 4'd0} : addr;
                m_arlen[8*ar_rr +: 8]       <= nbeat - 1;
                q_word[ar_rr][q_wp[ar_rr][2:0]] <= cur_word;
                q_lane[ar_rr][q_wp[ar_rr][2:0]] <= cur_lane;
                q_s0[ar_rr][q_wp[ar_rr][2:0]]   <= s0;
                q_tf[ar_rr][q_wp[ar_rr][2:0]]   <= tfull;
                q_wp[ar_rr]                 <= q_wp[ar_rr] + 1;
                issued                      <= issued + 1;
                ar_rr <= ar_rr == NPORTS - 1 ? 0 : ar_rr + 1;
                if (off + {nl, 3'b000} == row_bytes) begin         // next row
                    off     <= 0;
                    r       <= r + 1;
                    row_ddr <= row_ddr + pitch;
                    cur_lane <= 0;
                    if (step == 1) begin
                        row_word <= row_word + wpr;
                        cur_word <= row_word + wpr;
                    end else begin
                        row_word <= row_word + 1;
                        cur_word <= row_word + 1;
                    end
                    if (r + 1 == rows) gen <= 0;
                end else begin
                    off      <= off + {nl, 3'b000};
                    cur_word <= cur_word + words_adv * step;
                    cur_lane <= lane_sum[7:0] & lane_mask;
                end
            end else if (NPORTS > 1 && gen && (m_arvalid[ar_rr] || q_full[ar_rr])) begin
                ar_rr <= ar_rr == NPORTS - 1 ? 0 : ar_rr + 1;      // try the next port
            end

            if (active && !gen && completed == issued && !(cmd_valid && cmd_ready)) begin
                active <= 0;
                done   <= 1;
                err    <= rerr;
            end
            if (r_take && m_rresp[2*r_sel +: 2] != 2'b00) rerr <= 1;
        end
    end

    // ------------------------------------------------------------ R side
    // one local write per cycle, round-robin over ports that have data; a beat
    // written in two cycles keeps its port (r_hold)
    reg           r_hold;         // the selected port's beat: its upper lane next
    reg  [PB-1:0] r_hport;
    integer s, cand;
    always @* begin
        r_any = 0;
        r_sel = 0;
        if (r_hold) begin
            r_any = 1;
            r_sel = r_hport;
        end else
            for (s = NPORTS - 1; s >= 0; s = s - 1) begin
                cand = (r_rr + s) % NPORTS;
                if (m_rvalid[cand] && !q_empty[cand]) begin
                    r_any = 1;
                    r_sel = cand;
                end
            end
    end

    reg  [4:0] beat_idx [0:NPORTS-1];     // beats already taken from the head burst
    wire [15:0] h_word = q_word[r_sel][q_rp[r_sel][2:0]];
    wire [7:0]  h_lane = q_lane[r_sel][q_rp[r_sel][2:0]];
    wire        h_s0   = q_s0[r_sel][q_rp[r_sel][2:0]];
    wire        h_tf   = q_tf[r_sel][q_rp[r_sel][2:0]];
    wire        h_first = beat_idx[r_sel] == 0;
    wire        v0     = !(h_first && h_s0);                       // lower lane ours
    wire        v1     = W2 && !(m_rlast[r_sel] && !h_tf);         // upper lane ours
    // lane of this beat's lower lane in the burst's sequence (it is ours when v0)
    wire [8:0]  h_sum0 = h_lane + ({4'd0, beat_idx[r_sel]} << W2) - h_s0;
    wire        two    = v0 && v1;
    wire        merge  = two && lpwl != 0 && !h_sum0[0];          // both lanes, one word, even lane
    wire        upper  = r_hold || !v0;                            // writing the upper lane
    wire [8:0]  w_sum  = h_sum0 + upper;
    assign      r_take = r_any && !(two && !merge && !r_hold);     // the beat is consumed

    wire [DMA_W-1:0] r_beat = m_rdata[DMA_W*r_sel +: DMA_W];
    wire [63:0]      r_lane = upper ? r_beat[DMA_W-1 -: 64] : r_beat[63:0];

    assign lw_en   = r_any;
    assign lw_word = h_word + (w_sum >> lpwl) * step;
    assign lw_lane = w_sum[7:0] & lane_mask;
    assign lw_two  = merge;
    assign lw_data = merge ? r_beat : {BL{r_lane}};

    generate
        for (gp = 0; gp < NPORTS; gp = gp + 1) begin : rr
            assign m_rready[gp] = r_take && r_sel == gp;
        end
    endgenerate

    always @(posedge clk) begin
        if (!resetn) begin
            r_rr      <= 0;
            r_hold    <= 0;
            completed <= 0;
            for (p = 0; p < NPORTS; p = p + 1) begin
                q_rp[p]     <= 0;
                beat_idx[p] <= 0;
            end
        end else begin
            if (cmd_valid && cmd_ready) completed <= 0;
            if (r_any && !r_take) begin
                r_hold  <= 1;
                r_hport <= r_sel;
            end
            if (r_take) begin
                r_hold <= 0;
                r_rr   <= r_sel == NPORTS - 1 ? 0 : r_sel + 1;
                if (m_rlast[r_sel]) begin
                    beat_idx[r_sel] <= 0;
                    q_rp[r_sel]     <= q_rp[r_sel] + 1;
                    completed       <= completed + 1;
                end else
                    beat_idx[r_sel] <= beat_idx[r_sel] + 1;
            end
        end
    end

    assign perf_ev  = {|(m_arvalid & ~m_arready), lw_en, active};
    assign perf_two = lw_en && lw_two;
endmodule
