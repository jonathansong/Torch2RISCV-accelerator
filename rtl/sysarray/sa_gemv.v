// GEMV unit: the LD engine's mode GEMV (K2b, docs/k2b_gemv_design.md §4, §6).
//
// A command computes, for strips s < S (<= 8) and columns j < D,
//   ACC[C + s*cstep][j] = (acc ? ACC[C + s*cstep][j] : 0) + sum_k x[k] * W_s[k][j]
// (int8 x int8, int32 wrap-around); x = K int8 at SPAD_A word xword on, W_s
// streamed by sa_ld: one beat = one k row of D int8 weights of one strip,
// delivered on lane l (one lane per AXI read port) with its (s, k).
//
//   x buffer   per lane, K_MAX = 4096 bytes (256 words of D bytes). Loaded from
//              SPAD_A (one word per cycle) when a command starts, unless the
//              residency flag says it already holds [xword, xword + K/D): the
//              flag is cleared by any write to SPAD_A (x_inval), conservative.
//              Beats are held off (xready = 0) while it loads.
//   lanes      beat -> x word read (1) -> 16 products (2) -> accumulate (3);
//              per lane acc[s] (S_MAX x D x int32); a strip's first beat on a
//              lane overwrites instead of adding (touched mask), so no clear.
//   writeback  after all S * K beats: per strip, the lanes' acc[s] summed
//              (+ the ACC word, read first, when acc) and written to ACC.
//
// D = 16 only (a beat of the 128-bit DMA is one k row). The ACC port is the
// LD engine's (side B, requester 0), SPAD_A read is the LD write port (side
// A, requester 0); the LD engine runs one command at a time, so both are free.
`timescale 1ns / 1ps

module sa_gemv #(
    parameter integer D   = 16,
    parameter integer NL  = 1,                       // lanes (= AXI read ports)
    parameter integer SAW = 13,
    parameter integer CAW = 12
) (
    input  wire                 clk,
    input  wire                 resetn,

    // command: start pulses when the LD engine accepts a GEMV
    input  wire                 start,
    input  wire [15:0]          cmd_xword,
    input  wire [15:0]          cmd_c,
    input  wire [15:0]          cmd_strips,
    input  wire [15:0]          cmd_rb,              // K * D
    input  wire [7:0]           cmd_cstep,
    input  wire                 cmd_acc,
    output reg                  busy,                // start .. results written
    output wire                 xready,              // beats may be taken

    // weight beats (taken whenever valid: no back-pressure but xready)
    input  wire [NL-1:0]        b_valid,
    input  wire [NL*8*D-1:0]    b_data,
    input  wire [NL*3-1:0]      b_s,
    input  wire [NL*12-1:0]     b_k,

    input  wire                 x_inval,             // a write to SPAD_A this cycle

    // SPAD_A read (x), one cycle latency
    output wire                 xr_en,
    output wire [SAW-1:0]       xr_addr,
    input  wire [8*D-1:0]       xr_data,

    // ACC (results): read (acc) or write, one cycle read latency
    output wire                 c_en,
    output wire                 c_we,
    output wire [CAW-1:0]       c_addr,
    output wire [32*D-1:0]      c_din,
    input  wire [32*D-1:0]      c_dout
);
    localparam integer LOGD = $clog2(D);
    localparam integer SMAX = 8;

    // ---------------------------------------------------------- command
    reg  [15:0] xword, c_word;
    reg  [2:0]  s_last;
    reg  [7:0]  cstep;
    reg         acc;
    reg  [15:0] beats_left;                         // beats not yet accumulated
    wire [15:0] c_nb = (cmd_rb >> LOGD) * cmd_strips;   // K beats per strip
    wire [8:0]  c_nw = cmd_rb >> (2 * LOGD);            // x words (K / D <= 255)

    // ---------------------------------------------------------- x buffer
    reg         xv;                                 // residency flag
    reg  [15:0] xv_word;
    reg  [8:0]  xv_nw;
    reg         loading, dirty;
    reg  [8:0]  ld_i;                               // next word to read
    reg         ld_rv;                              // a read issued last cycle
    reg  [7:0]  ld_wi;                              // its buffer index
    wire        hit = xv && !x_inval && xv_word == cmd_xword && xv_nw >= c_nw;

    assign xready  = !loading;
    assign xr_en   = loading && ld_i != xv_nw;
    assign xr_addr = xword + ld_i;

    always @(posedge clk) begin
        if (!resetn) begin
            xv <= 0; loading <= 0; ld_rv <= 0;
        end else begin
            ld_rv <= xr_en;
            ld_wi <= ld_i[7:0];
            if (start) begin
                if (!hit) begin
                    loading <= 1; dirty <= 0; ld_i <= 0;
                    xv <= 0; xv_word <= cmd_xword; xv_nw <= c_nw;
                end
            end else if (loading) begin
                if (xr_en) ld_i <= ld_i + 1;
                if (x_inval) dirty <= 1;
                if (!xr_en && !ld_rv) begin             // last word written
                    loading <= 0;
                    xv      <= !dirty && !x_inval;
                end
            end else if (x_inval)
                xv <= 0;
        end
    end

    // ------------------------------------------------------------- lanes
    wire [NL-1:0]   l_done;                         // a beat accumulated this cycle
    wire [32*D-1:0] l_acc [0:NL-1];                 // acc[wb_s] of each lane (writeback)
    wire [NL-1:0]   l_tch;
    reg  [2:0]      wb_s;
    reg             wb;                             // writing back

    genvar l, j;
    generate
        for (l = 0; l < NL; l = l + 1) begin : lane
            reg [8*D-1:0] xb [0:255];
            always @(posedge clk) if (ld_rv) xb[ld_wi] <= xr_data;

            // stage 1: x word read
            reg           s1_v;
            reg [2:0]     s1_s;
            reg [3:0]     s1_b;
            reg [8*D-1:0] s1_w, s1_x;
            always @(posedge clk) begin
                s1_v <= resetn && b_valid[l] && xready;
                s1_s <= b_s[3*l +: 3];
                s1_b <= b_k[12*l +: 4];
                s1_w <= b_data[8*D*l +: 8*D];
                s1_x <= xb[b_k[12*l + 4 +: 8]];
            end
            // stage 2: products
            wire signed [7:0] xk = s1_x[8*s1_b +: 8];
            reg           s2_v;
            reg [2:0]     s2_s;
            reg [16*D-1:0] s2_p;
            for (j = 0; j < D; j = j + 1) begin : mul
                wire signed [7:0] w8 = s1_w[8*j +: 8];
                always @(posedge clk) s2_p[16*j +: 16] <= xk * w8;
            end
            always @(posedge clk) begin
                s2_v <= resetn && s1_v;
                s2_s <= s1_s;
            end
            // stage 3: accumulate (first beat of a strip on this lane: overwrite)
            reg [32*D-1:0] am [0:SMAX-1];
            reg [SMAX-1:0] tch;
            wire [2:0]     ra = wb ? wb_s : s2_s;
            wire [32*D-1:0] cur = am[ra];
            wire [32*D-1:0] nxt;
            for (j = 0; j < D; j = j + 1) begin : add
                assign nxt[32*j +: 32] = (tch[s2_s] ? cur[32*j +: 32] : 32'd0) +
                                         {{16{s2_p[16*j + 15]}}, s2_p[16*j +: 16]};
            end
            always @(posedge clk) begin
                if (s2_v) am[s2_s] <= nxt;
                if (start) tch <= 0;
                else if (s2_v) tch[s2_s] <= 1;
            end
            assign l_done[l] = s2_v;
            assign l_acc[l]  = cur;
            assign l_tch[l]  = tch[wb_s];
        end
    endgenerate

    // --------------------------------------------------------- writeback
    // per strip: (acc) read the ACC word, then write; else write only
    reg         wb_rd;                              // the ACC word was read last cycle
    reg  [15:0] wb_word;
    reg  [32*D-1:0] sum;
    integer q, n;
    always @* begin
        sum = 0;
        for (q = 0; q < D; q = q + 1) begin
            sum[32*q +: 32] = acc ? c_dout[32*q +: 32] : 32'd0;
            for (n = 0; n < NL; n = n + 1)
                if (l_tch[n]) sum[32*q +: 32] = sum[32*q +: 32] + l_acc[n][32*q +: 32];
        end
    end
    wire wb_write = wb && (!acc || wb_rd);
    assign c_en   = wb;
    assign c_we   = wb_write;
    assign c_addr = wb_word[CAW-1:0];
    assign c_din  = sum;

    integer cnt;
    always @* begin
        cnt = 0;
        for (n = 0; n < NL; n = n + 1) cnt = cnt + l_done[n];
    end

    always @(posedge clk) begin
        if (!resetn) begin
            busy <= 0; wb <= 0; wb_rd <= 0;
        end else begin
            if (start) begin
                busy       <= 1;
                xword      <= cmd_xword;
                c_word     <= cmd_c;
                s_last     <= cmd_strips[2:0] - 3'd1;
                cstep      <= cmd_cstep;
                acc        <= cmd_acc;
                beats_left <= c_nb;
            end else if (busy && !wb) begin
                beats_left <= beats_left - cnt;
                if (beats_left == cnt) begin            // the last beats accumulate now
                    wb      <= 1;
                    wb_s    <= 0;
                    wb_rd   <= 0;
                    wb_word <= c_word;
                end
            end else if (wb) begin
                wb_rd <= acc && !wb_rd;
                if (wb_write) begin
                    if (wb_s == s_last) begin
                        wb   <= 0;
                        busy <= 0;
                    end
                    wb_s    <= wb_s + 1;
                    wb_word <= wb_word + cstep;
                end
            end
        end
    end
endmodule
