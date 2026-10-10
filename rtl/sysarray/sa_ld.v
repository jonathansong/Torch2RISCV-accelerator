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
//
// Mode GEMV (K2b, docs/k2b_gemv_design.md §5; D = 16, DMA_W = 128): rows =
// strips, each row is K beats of one k row of weights. Nothing is written
// locally: each burst remembers its (strip, first k), and every port hands
// its beats to its own sa_gemv lane in the same cycle (g_valid / g_s / g_k),
// as long as the x buffer is loaded (g_xready). The command is done when all
// beats are taken and sa_gemv has written the results (!g_busy).
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

    // mode GEMV: beats to the sa_gemv lanes (one per port)
    input  wire                 g_xready,
    input  wire                 g_busy,
    output wire [NPORTS-1:0]    g_valid,
    output wire [NPORTS*DMA_W-1:0] g_data,
    output wire [NPORTS*3-1:0]  g_s,
    output wire [NPORTS*12-1:0] g_k,

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
    // perf_lanes: 8-byte lanes moved this cycle (LD_BEATS units)
    output wire [2:0]           perf_ev,
    output wire [3:0]           perf_lanes
);
    `include "sa_defs.vh"
    localparam integer LOGD = $clog2(D);
    localparam integer PB   = NPORTS > 1 ? $clog2(NPORTS) : 1;
    localparam integer QD   = 8;                      // outstanding bursts per port
    localparam integer BL   = DMA_W / 64;             // lanes per beat (1 or 2)
    localparam integer W2   = BL == 2;

    // R channel as seen by the logic below: with several ports, each port's R
    // goes through a 2-entry skid buffer first (registered valid / data /
    // ready: the PS R outputs then feed only flops, not the port arbitration
    // and the local write address); one port: straight through
    wire [NPORTS-1:0]       i_rvalid, i_rlast, i_rready;
    wire [NPORTS*DMA_W-1:0] i_rdata;
    wire [NPORTS*2-1:0]     i_rresp;
    genvar gs;
    generate
        if (NPORTS > 1) begin : rsl
            for (gs = 0; gs < NPORTS; gs = gs + 1) begin : port
                localparam integer RW = DMA_W + 3;
                reg [RW-1:0] d0, d1;                // output, skid
                reg          v0, v1;
                wire         push = m_rvalid[gs] && !v1;
                assign m_rready[gs] = !v1;
                assign i_rvalid[gs] = v0;
                assign {i_rlast[gs], i_rresp[2*gs +: 2], i_rdata[DMA_W*gs +: DMA_W]} = d0;
                always @(posedge clk) begin
                    if (!resetn) begin
                        v0 <= 0; v1 <= 0;
                    end else if (!v0 || i_rready[gs]) begin
                        if (v1) begin
                            d0 <= d1; v1 <= 0;
                        end else begin
                            v0 <= push;
                            d0 <= {m_rlast[gs], m_rresp[2*gs +: 2], m_rdata[DMA_W*gs +: DMA_W]};
                        end
                    end else if (push) begin
                        d1 <= {m_rlast[gs], m_rresp[2*gs +: 2], m_rdata[DMA_W*gs +: DMA_W]};
                        v1 <= 1;
                    end
                end
            end
        end else begin : rdirect
            assign i_rvalid = m_rvalid;
            assign i_rlast  = m_rlast;
            assign i_rresp  = m_rresp;
            assign i_rdata  = m_rdata;
            assign m_rready = i_rready;
        end
    endgenerate

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
    reg         gmode;            // the command is a GEMV
    wire [NPORTS-1:0] g_take;     // GEMV: beats taken this cycle, by port

    assign cmd_ready = !active && resetn;

    wire [3:0] c_lpwl = lpw_log2(cmd_mem);
    wire       flat   = cmd_mode == 2'd0 && cmd_pitch == cmd_row_bytes &&
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
    reg  [2:0]  q_s     [0:NPORTS-1][0:QD-1];    // GEMV: strip, first k
    reg  [11:0] q_k     [0:NPORTS-1][0:QD-1];
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
                gmode     <= cmd_mode == MODE_GEMV;
            end else if (issue_ok) begin
                m_arvalid[ar_rr]            <= 1;
                m_araddr[32*ar_rr +: 32]    <= W2 ? {addr[31:4], 4'd0} : addr;
                m_arlen[8*ar_rr +: 8]       <= nbeat - 1;
                q_word[ar_rr][q_wp[ar_rr][2:0]] <= cur_word;
                q_lane[ar_rr][q_wp[ar_rr][2:0]] <= cur_lane;
                q_s0[ar_rr][q_wp[ar_rr][2:0]]   <= s0;
                q_tf[ar_rr][q_wp[ar_rr][2:0]]   <= tfull;
                q_s[ar_rr][q_wp[ar_rr][2:0]]    <= r[2:0];
                q_k[ar_rr][q_wp[ar_rr][2:0]]    <= off[15:4];
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

            if (active && !gen && completed == issued && !g_busy && !(cmd_valid && cmd_ready)) begin
                active <= 0;
                done   <= 1;
                err    <= rerr;
            end
            if (r_take && i_rresp[2*r_sel +: 2] != 2'b00) rerr <= 1;
            for (p = 0; p < NPORTS; p = p + 1)
                if (g_take[p] && i_rresp[2*p +: 2] != 2'b00) rerr <= 1;
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
                if (!gmode && i_rvalid[cand] && !q_empty[cand]) begin
                    r_any = 1;
                    r_sel = cand;
                end
            end
    end

    reg  [4:0] beat_idx [0:NPORTS-1];     // beats already taken from the head burst

    // mode GEMV: every port with a beat hands it to its lane, all in one cycle
    generate
        for (gp = 0; gp < NPORTS; gp = gp + 1) begin : gq
            wire [2:0] hp = q_rp[gp][2:0];
            assign g_take[gp] = gmode && active && g_xready && i_rvalid[gp] && !q_empty[gp];
            assign g_s[3*gp +: 3]    = q_s[gp][hp];
            assign g_k[12*gp +: 12]  = q_k[gp][hp] + beat_idx[gp];
        end
    endgenerate
    assign g_valid = g_take;
    assign g_data  = i_rdata;
    wire [15:0] h_word = q_word[r_sel][q_rp[r_sel][2:0]];
    wire [7:0]  h_lane = q_lane[r_sel][q_rp[r_sel][2:0]];
    wire        h_s0   = q_s0[r_sel][q_rp[r_sel][2:0]];
    wire        h_tf   = q_tf[r_sel][q_rp[r_sel][2:0]];
    wire        h_first = beat_idx[r_sel] == 0;
    wire        v0     = !(h_first && h_s0);                       // lower lane ours
    wire        v1     = W2 && !(i_rlast[r_sel] && !h_tf);         // upper lane ours
    // lane of this beat's lower lane in the burst's sequence (it is ours when v0)
    wire [8:0]  h_sum0 = h_lane + ({4'd0, beat_idx[r_sel]} << W2) - h_s0;
    wire        two    = v0 && v1;
    wire        merge  = two && lpwl != 0 && !h_sum0[0];          // both lanes, one word, even lane
    wire        upper  = r_hold || !v0;                            // writing the upper lane
    wire [8:0]  w_sum  = h_sum0 + upper;
    assign      r_take = r_any && !(two && !merge && !r_hold);     // the beat is consumed

    wire [DMA_W-1:0] r_beat = i_rdata[DMA_W*r_sel +: DMA_W];
    wire [63:0]      r_lane = upper ? r_beat[DMA_W-1 -: 64] : r_beat[63:0];

    wire             c_lw_en   = r_any;
    wire [15:0]      c_lw_word = h_word + (w_sum >> lpwl) * step;
    wire [7:0]       c_lw_lane = w_sum[7:0] & lane_mask;
    wire             c_lw_two  = merge;
    wire [DMA_W-1:0] c_lw_data = merge ? r_beat : {BL{r_lane}};

    // local write: NPORTS > 2 registers it (the port arbitration and the word
    // multiply then end at flops, not at the memories' address pins); the
    // command's done comes a cycle after its last beat is taken, with the
    // write already in this register, so it is written by then
    generate
        if (NPORTS > 2) begin : lwreg
            reg             r_en, r_two;
            reg [15:0]      r_word;
            reg [7:0]       r_lane;
            reg [DMA_W-1:0] r_data;
            always @(posedge clk) begin
                r_en   <= resetn && c_lw_en;
                r_word <= c_lw_word;
                r_lane <= c_lw_lane;
                r_two  <= c_lw_two;
                r_data <= c_lw_data;
            end
            assign lw_en = r_en; assign lw_word = r_word; assign lw_lane = r_lane;
            assign lw_two = r_two; assign lw_data = r_data;
        end else begin : lwdirect
            assign lw_en = c_lw_en; assign lw_word = c_lw_word; assign lw_lane = c_lw_lane;
            assign lw_two = c_lw_two; assign lw_data = c_lw_data;
        end
    endgenerate

    generate
        for (gp = 0; gp < NPORTS; gp = gp + 1) begin : rr
            assign i_rready[gp] = (r_take && r_sel == gp) || g_take[gp];
        end
    endgenerate

    integer g_last, gl;           // GEMV: bursts finished this cycle
    always @* begin
        g_last = 0;
        for (gl = 0; gl < NPORTS; gl = gl + 1) g_last = g_last + (g_take[gl] && i_rlast[gl]);
    end

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
            else if (gmode) completed <= completed + g_last;
            for (p = 0; p < NPORTS; p = p + 1)
                if (g_take[p]) begin
                    if (i_rlast[p]) begin
                        beat_idx[p] <= 0;
                        q_rp[p]     <= q_rp[p] + 1;
                    end else
                        beat_idx[p] <= beat_idx[p] + 1;
                end
            if (r_any && !r_take) begin
                r_hold  <= 1;
                r_hport <= r_sel;
            end
            if (r_take) begin
                r_hold <= 0;
                r_rr   <= r_sel == NPORTS - 1 ? 0 : r_sel + 1;
                if (i_rlast[r_sel]) begin
                    beat_idx[r_sel] <= 0;
                    q_rp[r_sel]     <= q_rp[r_sel] + 1;
                    completed       <= completed + 1;
                end else
                    beat_idx[r_sel] <= beat_idx[r_sel] + 1;
            end
        end
    end

    // GEMV beats count as local writes (BL lanes each, all ports)
    integer g_n, gc;
    always @* begin
        g_n = 0;
        for (gc = 0; gc < NPORTS; gc = gc + 1) g_n = g_n + g_take[gc];
    end
    assign perf_ev    = {|(m_arvalid & ~m_arready), lw_en || |g_take, active};
    assign perf_lanes = (lw_en ? (lw_two ? 4'd2 : 4'd1) : 4'd0) + BL * g_n;
endmodule
