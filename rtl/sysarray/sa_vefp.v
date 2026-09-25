// L2 fp32 vector engine and TRANSPOSE (docs/llm_inference_plan.md §6).
// Bit-exact with llm/sa_funcsim.py ve_fp() / ve_transpose() (fp32 rules:
// llm/fp32.py; special functions: llm/sfu.py).
//
// A command processes G groups of D lanes. Per group g, lane l:
//   a = src1[i1(g)][l], b = src2[i2(g)][l] or IMM   (index: LIN g, MOD g mod P, DIV g div P)
//   inputs converted to fp32 (I8 in SPAD, I32 / F32 in ACC), [SWAPNEG on a]
//   y = OP(a, b)  (ADD SUB MUL MAX MIN COPY),  y = y * A + B
//   y = FUNC(y)   (NONE EXP RECIP RSQRT ABS),  RELU,  VALID mask
//   output F32 / I32 (ACC) or I8 (SPAD, round to nearest even, +-127),
//   or REDUCE SUM / MAX per row (ROWLEN groups; lane-sequential, then a
//   pairwise lane tree; one broadcast word per row).
//
// Lane folding: FL = D / 2 physical fp lanes (parameter; FL = D also
// works). A group of D lanes goes through them as D / FL halves on
// consecutive cycles; the results are reassembled into one word. Reads
// already take 2 cycles per group for binary ops, so only unary ops and
// the sequenced mode run slower; the fp units, the sequencer registers and
// their operand multiplexers are halved (docs/llm_inference_plan.md §6.9).
//
// Datapath per physical lane: i2f x 2, adder A1 + multiplier M1 (OP),
// M2 + A2 (affine), f2i (output). Two modes:
//   stream     FUNC NONE / ABS, no REDUCE: a fixed-latency pipeline, one
//              half issued per cycle (reads permitting)
//   sequenced  EXP / RECIP / RSQRT or REDUCE: one group in flight; after A2
//              a microsequencer runs the special function half by half on
//              the lanes' M2 / A2 / i2f / f2i (llm/sfu.py step by step; the
//              segment tables are one ROM read lane by lane), then RELU /
//              VALID, the reduction (A2 or a comparator) or the output
//              conversion over the whole group.
// TRANSPOSE (op 6): per D x D block, D words read at stride S into a
// register buffer, D transposed words written.
// Memory ports as sa_ve (one per memory, writes have priority over reads);
// the idle engine of the pair drives zeros (ports are ORed in sa_vex).
`timescale 1ns / 1ps
`include "sa_macros.vh"

module sa_vefp #(
    parameter integer D       = 8,
    parameter integer FL      = D / 2,         // physical fp lanes: D / 2 or D
    parameter integer SPAD_AW = 14,
    parameter integer ACC_AW  = 13
) (
    input  wire                 clk,
    input  wire                 resetn,

    input  wire                 cmd_valid,
    output wire                 cmd_ready,
    input  wire [31:0]          cmd_src1,
    input  wire [31:0]          cmd_src2,
    input  wire [31:0]          cmd_dst,
    input  wire [15:0]          cmd_groups,
    input  wire [7:0]           cmd_op,
    input  wire [5:0]           cmd_types,
    input  wire [15:0]          cmd_mod,          // P2
    input  wire [10:0]          cmd_flags,        // [0] FP [3:1] FUNC [5:4] M1 [7:6] M2 [9:8] REDUCE [10] SWAPNEG
    input  wire [31:0]          cmd_imm,
    input  wire [31:0]          cmd_a,
    input  wire [31:0]          cmd_b,
    input  wire [15:0]          cmd_rowlen,
    input  wire [15:0]          cmd_vld,
    input  wire [15:0]          cmd_p1,
    input  wire [15:0]          cmd_s,
    output reg                  done,
    output wire                 busy,

    output reg                  sa_en,
    output reg  [D-1:0]         sa_we,
    output reg  [SPAD_AW-1:0]   sa_addr,
    output reg  [8*D-1:0]       sa_din,
    input  wire [8*D-1:0]       sa_dout,
    output reg                  sb_en,
    output reg  [D-1:0]         sb_we,
    output reg  [SPAD_AW-1:0]   sb_addr,
    output reg  [8*D-1:0]       sb_din,
    input  wire [8*D-1:0]       sb_dout,
    output reg                  ac_en,
    output reg  [4*D-1:0]       ac_we,
    output reg  [ACC_AW-1:0]    ac_addr,
    output reg  [32*D-1:0]      ac_din,
    input  wire [32*D-1:0]      ac_dout,

    // performance events: {group written, sequencer busy, read lost the port, command active}
    output wire [3:0]           perf_ev
);
    `include "sa_defs.vh"
    `include "sa_sfu_tables.vh"
    localparam integer LOGD = $clog2(D);
    localparam integer NH   = D / FL;             // halves per group (1 or 2)
    localparam [1:0] T_I8 = 2'd0, T_I32 = 2'd2, T_F32 = 2'd3;
    localparam [2:0] F_NONE = 3'd0, F_EXP = 3'd1, F_RECIP = 3'd2, F_RSQRT = 3'd3, F_ABS = 3'd4;
    localparam [1:0] X_LIN = 2'd0, X_MOD = 2'd1, X_DIV = 2'd2, X_IMM = 2'd3;
    localparam [31:0] QNAN = 32'h7FC0_0000, PINF = 32'h7F80_0000, NINF = 32'hFF80_0000;
    localparam [31:0] C_LOG2E = 32'h3FB8AA3B, C_64 = 32'h42800000, C_2 = 32'h40000000,
                      C_1_5 = 32'h3FC00000, C_0_5 = 32'h3F000000;
    localparam integer OQ = 32;                   // output FIFO (>= stream latency + reads)

    // ---------------------------------------------------------- command
    reg         active, xpose;
    reg  [3:0]  m1, m2, md;
    reg  [15:0] b1, b2, bd;
    reg  [15:0] G, rowlen, vld, P1, P2, S;
    reg  [2:0]  op;
    reg         relu, unary;
    reg  [1:0]  it, ot, it2;                 // it2: src2 type (types[5:4]: 0 = as src1, 1 I8, 2 I32, 3 F32)
    reg  [2:0]  func;
    reg  [1:0]  mo1, mo2, red;
    reg         swapneg, seqmode;
    reg  [31:0] imm, cA, cB;
    assign cmd_ready = !active && resetn;
    assign busy = active;

    // --------------------------------------------------- output FIFO / write
    reg  [32*D-1:0]  oq_d [0:OQ-1];
    reg  [15:0]      oq_a [0:OQ-1];
    reg  [5:0]       oq_wp, oq_rp;
    wire             oq_empty = oq_wp == oq_rp;
    wire             wr_now = !oq_empty;
    wire [3:0]       wmem = md;
    reg              push;
    reg  [32*D-1:0]  push_d;
    reg  [15:0]      push_a;

    // transposed words are written straight from the buffer (TRANSPOSE state machine)
    reg              tw_en;
    reg  [15:0]      tw_addr;
    reg  [32*D-1:0]  tw_d;

    // ---------------------------------------------------------- read unit
    reg  [15:0] rg;                  // groups started
    reg  [15:0] i1c, i1d, i2c, i2d;  // MOD counter / DIV counter per source
    reg  [15:0] gr, row;             // group within its row, row
    reg         rpart;               // 0: src1 read issued next, 1: src2
    reg  [6:0]  inflight;            // groups started, not retired
    wire [15:0] idx1 = mo1 == X_LIN ? rg : mo1 == X_MOD ? i1c : i1d;
    wire [15:0] idx2 = mo2 == X_LIN ? rg : mo2 == X_MOD ? i2c : i2d;
    wire        need2 = !unary && mo2 != X_IMM;
    wire [6:0]  cap = seqmode ? 7'd1 : OQ - 4;
    reg  [1:0]  rd_gap;                  // cycles until the next group's first read (NH > 1)
    wire        rd_can = active && !xpose && rg < G && (rpart || (inflight < cap && rd_gap == 0));
    wire [3:0]  rd_mem = rpart ? m2 : m1;
    wire [15:0] rd_word = rpart ? b2 + idx2 : b1 + idx1;
    wire        tr_rd;                      // TRANSPOSE read (below)
    wire [3:0]  tr_mem;
    wire [15:0] tr_word;
    wire        rd_block = (wr_now && wmem == rd_mem) || tw_en;
    wire        rd_issue = rd_can && !rd_block;
    wire        rd_last  = rpart || !need2;

    // memory ports (writes first)
    always @* begin
        sa_en = 0; sa_we = 0; sa_addr = 0; sa_din = 0;
        sb_en = 0; sb_we = 0; sb_addr = 0; sb_din = 0;
        ac_en = 0; ac_we = 0; ac_addr = 0; ac_din = 0;
        if (wr_now || tw_en) begin : wr
            reg [32*D-1:0] wd_;
            reg [15:0]     wa_;
            integer q;
            wd_ = tw_en ? tw_d : oq_d[oq_rp[4:0]];
            wa_ = tw_en ? tw_addr : oq_a[oq_rp[4:0]];
            case (wmem)
                MEM_SPAD_A: begin sa_en = 1; sa_we = {D{1'b1}}; sa_addr = wa_[SPAD_AW-1:0];
                                  for (q = 0; q < D; q = q + 1) sa_din[8*q +: 8] = tw_en ? wd_[8*q +: 8] : wd_[32*q +: 8]; end
                MEM_SPAD_B: begin sb_en = 1; sb_we = {D{1'b1}}; sb_addr = wa_[SPAD_AW-1:0];
                                  for (q = 0; q < D; q = q + 1) sb_din[8*q +: 8] = tw_en ? wd_[8*q +: 8] : wd_[32*q +: 8]; end
                default:    begin ac_en = 1; ac_we = {4*D{1'b1}}; ac_addr = wa_[ACC_AW-1:0]; ac_din = wd_; end
            endcase
        end
        if (rd_issue || tr_rd) begin : rd
            reg [3:0]  rm_;
            reg [15:0] rw_;
            rm_ = tr_rd ? tr_mem : rd_mem;
            rw_ = tr_rd ? tr_word : rd_word;
            case (rm_)
                MEM_SPAD_A: begin sa_en = 1; sa_addr = rw_[SPAD_AW-1:0]; end
                MEM_SPAD_B: begin sb_en = 1; sb_addr = rw_[SPAD_AW-1:0]; end
                default:    begin ac_en = 1; ac_addr = rw_[ACC_AW-1:0]; end
            endcase
        end
    end

    // read return
    reg         ret_v, ret_src2, ret_last, ret_first, ret_rowlast;
    reg  [3:0]  ret_mem;
    reg  [D-1:0] ret_mask;
    reg  [15:0] ret_dst;
    wire [32*D-1:0] ret_data = ret_mem == MEM_ACC ? ac_dout :
                               ret_mem == MEM_SPAD_B ? {{24*D{1'b0}}, sb_dout} : {{24*D{1'b0}}, sa_dout};

    // per-group tag assembled at issue: VALID mask, first / last of row, destination word
    reg  [D-1:0] g_mask;
    integer ml;
    always @* for (ml = 0; ml < D; ml = ml + 1) g_mask[ml] = vld == 0 || ({gr, {LOGD{1'b0}}} + ml) < vld;

    // ------------------------------------------------------ lane datapath
    // stage P0: raw operand words (+ tag) of a complete group
    reg              p0_v;
    reg  [32*D-1:0]  rawa, rawb;
    reg  [D-1:0]     p0_gmask;              // VALID mask of the whole group
    reg              p0_h;                  // half at P0
    reg              p0_more;               // another half follows
    reg              p0_first, p0_last;
    reg  [15:0]      p0_dst;
    wire [FL-1:0]    p0_mask = p0_gmask >> (p0_h * FL);
    function [31:0] lane_raw(input [32*D-1:0] w, input [1:0] t, input integer l);
        lane_raw = t == T_I8 ? {{24{w[8*l+7]}}, w[8*l +: 8]} : w[32*l +: 32];
    endfunction
    function [31:0] ftz(input [31:0] x);
        ftz = x[30:23] == 0 ? {x[31], 31'd0} : x;
    endfunction

    // control pipeline (valid + tag), fixed latencies: I2F 2, OP 4, M2 4, A2 4
    localparam integer PL = 16;             // P0 .. P15 (A2 result at P15)
    reg  [PL:0]  pv;
    reg  [FL-1:0] pmask [0:PL];
    reg  [PL:0]  pfirst, plast, ph, phl;       // + half, last half of the group
    reg  [15:0]  pdst  [0:PL];

    // sequencer interface to the shared units
    reg          sq_act;                    // the sequencer owns M2 / A2 / i2f(a) / f2i
    reg          sq_m_v, sq_a_v, sq_i_v, sq_f_v, sq_a_sub, sq_f_floor;
    reg  [31:0]  sq_f_lo, sq_f_hi;
    reg  [31:0]  sq_ma [0:FL-1], sq_mb [0:FL-1], sq_aa [0:FL-1], sq_ab [0:FL-1], sq_ix [0:FL-1], sq_fx [0:FL-1];

    wire [31:0]  fa [0:FL-1], fb [0:FL-1];  // converted operands (P2)
    wire [31:0]  fsw [0:FL-1];              // SWAPNEG of fa: (x0, x1) -> (-x1, x0)
    wire [31:0]  ia_y_arr [0:FL-1];         // i2f of operand a (shared with the sequencer)
    wire [31:0]  opr_add [0:FL-1], opr_mul [0:FL-1];
    wire [31:0]  m2o [0:FL-1], a2o [0:FL-1], f2o [0:FL-1];
    reg  [31:0]  opA [0:FL-1], opB [0:FL-1];   // P3: OP inputs
    reg  [31:0]  cmp1 [0:FL-1], cmp2 [0:FL-1], cmp3 [0:FL-1], cmp4 [0:FL-1];   // MAX / MIN / COPY delay line
    reg  [31:0]  y7 [0:FL-1];               // P7: OP result (register), M2 input
    reg  [31:0]  y16 [0:FL-1];              // stream: after FUNC / RELU / mask
    reg          fo_isint;                  // output converted by f2i

    // fp compare helpers
    function lt_f(input [31:0] x, input [31:0] y);       // x < y (neither NaN; +-0 equal)
        reg zx, zy;
        begin
            zx = x[30:0] == 0; zy = y[30:0] == 0;
            if (zx && zy) lt_f = 0;
            else if (x[31] != y[31]) lt_f = x[31];
            else if (!x[31]) lt_f = x[30:0] < y[30:0];
            else lt_f = x[30:0] > y[30:0];
        end
    endfunction
    function is_nan(input [31:0] x); is_nan = x[30:23] == 8'hFF && x[22:0] != 0; endfunction
    function [31:0] fmax(input [31:0] x, input [31:0] y);
        fmax = is_nan(x) || is_nan(y) ? QNAN : lt_f(y, x) ? x : lt_f(x, y) ? y : (x & y);
    endfunction
    function [31:0] fmin(input [31:0] x, input [31:0] y);
        fmin = is_nan(x) || is_nan(y) ? QNAN : lt_f(x, y) ? x : lt_f(y, x) ? y : (x | y);
    endfunction
    function [31:0] relu_f(input [31:0] x, input en);
        relu_f = en && x[31] && !is_nan(x) ? 32'd0 : x;
    endfunction
    // x * 2^n (x normal, zero, inf or NaN; FTZ / overflow)
    function [31:0] ldexp(input [31:0] x, input signed [11:0] n);
        reg signed [12:0] e;
        begin
            e = $signed({5'd0, x[30:23]}) + n;
            if (is_nan(x)) ldexp = QNAN;
            else if (x[30:23] == 8'hFF || x[30:23] == 0) ldexp = x[30:23] == 0 ? {x[31], 31'd0} : x;
            else if (e < 1) ldexp = {x[31], 31'd0};
            else if (e > 254) ldexp = {x[31], 8'hFF, 23'd0};
            else ldexp = {x[31], e[7:0], x[22:0]};
        end
    endfunction

    genvar gl;
    generate
        for (gl = 0; gl < FL; gl = gl + 1) begin : lane
            wire [31:0] ra = lane_raw(rawa, it, p0_h * FL + gl), rb = lane_raw(rawb, it2, p0_h * FL + gl);
            wire [31:0] ia_x = sq_act ? sq_ix[gl] : ra;
            wire [31:0] ia_y, ib_y;
            sa_fp32_i2f u_ia (.clk(clk), .in_v(1'b1), .x(ia_x), .in_tag(1'b0), .out_v(), .y(ia_y), .out_tag());
            sa_fp32_i2f u_ib (.clk(clk), .in_v(1'b1), .x(rb), .in_tag(1'b0), .out_v(), .y(ib_y), .out_tag());
            reg [31:0] fa1, fa2, fb1, fb2;          // F32 path: FTZ, delayed like i2f
            always @(posedge clk) begin
                fa1 <= ftz(ra); fa2 <= fa1;
                fb1 <= ftz(rb); fb2 <= fb1;
            end
            assign fa[gl] = it == T_F32 ? fa2 : ia_y;
            assign ia_y_arr[gl] = ia_y;
            if (gl % 2 == 0) begin : ev
                assign fsw[gl] = fa[gl + 1] ^ 32'h8000_0000;
            end else begin : od
                assign fsw[gl] = fa[gl - 1];
            end
            assign fb[gl] = mo2 == X_IMM ? imm : it2 == T_F32 ? fb2 : ib_y;
            sa_fp32_add u_a1 (.clk(clk), .in_v(1'b1), .a(opA[gl]), .b(opB[gl]), .sub(op == VOP_SUB),
                              .in_tag(1'b0), .out_v(), .y(opr_add[gl]), .out_tag());
            sa_fp32_mul u_m1 (.clk(clk), .in_v(1'b1), .a(opA[gl]), .b(opB[gl]), .in_tag(1'b0), .out_v(),
                              .y(opr_mul[gl]), .out_tag());
            sa_fp32_mul u_m2 (.clk(clk), .in_v(1'b1), .a(sq_act ? sq_ma[gl] : y7[gl]),
                              .b(sq_act ? sq_mb[gl] : cA), .in_tag(1'b0), .out_v(), .y(m2o[gl]), .out_tag());
            sa_fp32_add u_a2 (.clk(clk), .in_v(1'b1), .a(sq_act ? sq_aa[gl] : m2o[gl]),
                              .b(sq_act ? sq_ab[gl] : cB), .sub(sq_act && sq_a_sub),
                              .in_tag(1'b0), .out_v(), .y(a2o[gl]), .out_tag());
            sa_fp32_f2i u_f (.clk(clk), .in_v(1'b1), .x(sq_act ? sq_fx[gl] : y16[gl]),
                             .floor_mode(sq_act && sq_f_floor),
                             .lo(sq_act ? sq_f_lo : ot == T_I8 ? -32'sd127 : 32'h8000_0000),
                             .hi(sq_act ? sq_f_hi : ot == T_I8 ? 32'sd127 : 32'h7FFF_FFFF),
                             .in_tag(1'b0), .out_v(), .y(f2o[gl]), .out_tag());
        end
    endgenerate

    // stream-mode output value of a lane (P15 -> P16): FUNC ABS / NONE, RELU, VALID
    integer l;
    always @(posedge clk) begin
        // P2 -> P3: SWAPNEG, COPY
        for (l = 0; l < FL; l = l + 1) begin
            opA[l] <= swapneg ? fsw[l] : fa[l];
            opB[l] <= fb[l];
        end
        // P3 -> P7: MAX / MIN / COPY through a delay line
        for (l = 0; l < FL; l = l + 1) begin
            cmp1[l] <= op == VOP_MAX ? fmax(opA[l], opB[l]) : op == VOP_MIN ? fmin(opA[l], opB[l]) : opA[l];
            cmp2[l] <= cmp1[l]; cmp3[l] <= cmp2[l]; cmp4[l] <= cmp3[l];
        end
        // P7: OP result
        for (l = 0; l < FL; l = l + 1)
            y7[l] <= op == VOP_ADD || op == VOP_SUB ? opr_add[l] : op == VOP_MUL ? opr_mul[l] : cmp4[l];
        // P15 -> P16 (stream)
        for (l = 0; l < FL; l = l + 1)
            y16[l] <= pmask[PL-1][l] ? relu_f(func == F_ABS ? {1'b0, a2o[l][30:0]} : a2o[l], relu) : 32'd0;
    end

    // ------------------------------------------------ control pipeline
    // pv[k]: group at stage k. P0 = operands captured; P2 = converted; P3 = OP
    // inputs; P7 = OP result; P11 = M2 result; P15 = A2 result.
    // stream: P16 = y16, P18 = f2i result -> FIFO. sequenced: P15 -> sequencer.
    reg  [2:0]  sp_v, sp_h, sp_hl;            // P16, P17, P18
    reg  [15:0] sp_dst [0:2];
    integer k;
    always @(posedge clk) begin
        if (!resetn) begin
            pv <= 0; sp_v <= 0;
        end else begin
            pv[0] <= p0_v;
            pmask[0] <= p0_mask; pfirst[0] <= p0_first; plast[0] <= p0_last; pdst[0] <= p0_dst;
            ph[0] <= p0_h; phl[0] <= !p0_more;
            for (k = 1; k <= PL; k = k + 1) begin
                pv[k] <= pv[k-1]; pmask[k] <= pmask[k-1]; pfirst[k] <= pfirst[k-1];
                plast[k] <= plast[k-1]; pdst[k] <= pdst[k-1]; ph[k] <= ph[k-1]; phl[k] <= phl[k-1];
            end
            sp_v[0] <= pv[PL-1] && !seqmode;
            sp_dst[0] <= pdst[PL-1]; sp_h[0] <= ph[PL-1]; sp_hl[0] <= phl[PL-1];
            sp_v[1] <= sp_v[0]; sp_dst[1] <= sp_dst[0]; sp_h[1] <= sp_h[0]; sp_hl[1] <= sp_hl[0];
            sp_v[2] <= sp_v[1]; sp_dst[2] <= sp_dst[1]; sp_h[2] <= sp_h[1]; sp_hl[2] <= sp_hl[1];
        end
    end
    // stream result halves, reassembled into group words
    reg [32*FL-1:0] y17, y18;
    always @(posedge clk) begin
        for (l = 0; l < FL; l = l + 1) y17[32*l +: 32] <= y16[l];
        y18 <= y17;
    end
    wire [32*FL-1:0] f2o_w;
    generate for (gl = 0; gl < FL; gl = gl + 1) begin : fw assign f2o_w[32*gl +: 32] = f2o[gl]; end endgenerate
    wire [32*FL-1:0] sp_half = ot == T_F32 ? y18 : f2o_w;
    reg  [32*D-1:0]  sp_asm;                  // halves of the group so far
    wire [32*D-1:0]  sp_word = NH == 1 ? sp_half : sp_h[2] ? {sp_half, sp_asm[32*FL-1:0]} : {sp_asm[32*D-1:32*FL], sp_half};
    always @(posedge clk) if (sp_v[2]) sp_asm <= sp_word;

    // ------------------------------------------------------- sequencer
    // micro-op units: 1 MUL (M2), 2 ADD (A2), 3 SUB (A2), 4 I2F (i2f a), 5 F2I floor (full range),
    // 6 F2I floor 0..63, 7 LOOKUP (serial over lanes), 8 x 2^-17, 9 END (function result)
    // operand codes: 0..7 RF, 8 LOG2E, 9 64.0, 10 2.0, 11 1.5, 12 0.5, 13 T, 14 S, 15 per-function input
    reg  [31:0] RF [0:FL-1][0:7];           // per physical lane (the half being evaluated)
    reg  [31:0] TT [0:FL-1], SS [0:FL-1];
    reg  [31:0] SQIN [0:D-1];               // A2 results of the group's halves
    reg  [31:0] VV [0:D-1];                 // function result / value after RELU + mask (whole group)
    reg  [31:0] ACCR [0:D-1];               // reduction accumulators
    reg         sq_h;                       // half being evaluated / accumulated / converted
    reg  [4:0]  sq_st;                      // main state
    reg  [4:0]  sq_step;
    reg  [3:0]  sq_wait;
    reg  [LOGD:0] sq_lane;
    reg  [3:0]  sq_lvl;                     // reduction tree level
    reg         sq_first, sq_last;
    reg  [D-1:0] sq_mask;
    reg  [15:0] sq_dst;
    localparam [4:0] Q_IDLE = 0, Q_ISSUE = 1, Q_WAIT = 2, Q_LOOK = 3, Q_POST = 4, Q_RED = 5, Q_REDW = 6,
                     Q_TREE = 7, Q_TREEW = 8, Q_OUT = 9, Q_OUTW = 10, Q_FIN = 11, Q_LOAD = 12;

    // micro-program: {unit[3:0], a[3:0], b[3:0], dst[2:0]}
    reg [14:0] uop;
    always @* begin
        uop = 15'd0;
        case (func)
            F_EXP: case (sq_step)
                0:  uop = {4'd1, 4'd0, 4'd8, 3'd1};     // t  = x * log2e
                1:  uop = {4'd5, 4'd1, 4'd0, 3'd2};     // n  = floor(t)
                2:  uop = {4'd4, 4'd2, 4'd0, 3'd3};     // nf = float(n)
                3:  uop = {4'd3, 4'd1, 4'd3, 3'd3};     // f  = t - nf
                4:  uop = {4'd1, 4'd3, 4'd9, 3'd4};     // u  = f * 64
                5:  uop = {4'd6, 4'd4, 4'd0, 3'd5};     // i  = floor(u), 0..63
                6:  uop = {4'd4, 4'd5, 4'd0, 3'd6};     // if = float(i)
                7:  uop = {4'd3, 4'd4, 4'd6, 3'd6};     // r  = u - if
                8:  uop = {4'd7, 4'd5, 4'd0, 3'd0};     // T, S [i]
                9:  uop = {4'd1, 4'd6, 4'd14, 3'd6};    // p  = r * S
                10: uop = {4'd2, 4'd13, 4'd6, 3'd7};    // y  = T + p
                default: uop = {4'd9, 11'd0};
            endcase
            F_RECIP: case (sq_step)
                0:  uop = {4'd4, 4'd15, 4'd0, 3'd1};    // ri = float(frac & 0x1FFFF)
                1:  uop = {4'd8, 4'd1, 4'd0, 3'd1};     // r  = ri * 2^-17
                2:  uop = {4'd7, 4'd0, 4'd0, 3'd0};     // T, S [frac >> 17]
                3:  uop = {4'd1, 4'd1, 4'd14, 3'd2};    // p  = r * S
                4:  uop = {4'd2, 4'd13, 4'd2, 3'd3};    // y0 = T + p
                5:  uop = {4'd1, 4'd15, 4'd3, 3'd4};    // t  = m * y0
                6:  uop = {4'd3, 4'd10, 4'd4, 3'd4};    // e  = 2 - t
                7:  uop = {4'd1, 4'd3, 4'd4, 3'd5};     // y1 = y0 * e
                default: uop = {4'd9, 11'd0};
            endcase
            F_RSQRT: case (sq_step)
                0:  uop = {4'd4, 4'd15, 4'd0, 3'd1};
                1:  uop = {4'd8, 4'd1, 4'd0, 3'd1};
                2:  uop = {4'd7, 4'd0, 4'd0, 3'd0};     // T, S [odd * 64 + frac >> 17]
                3:  uop = {4'd1, 4'd1, 4'd14, 3'd2};    // p  = r * S
                4:  uop = {4'd2, 4'd13, 4'd2, 3'd3};    // y0 = T + p
                5:  uop = {4'd1, 4'd3, 4'd3, 3'd4};     // a  = y0 * y0
                6:  uop = {4'd1, 4'd15, 4'd4, 3'd4};    // b  = m' * a
                7:  uop = {4'd1, 4'd4, 4'd12, 3'd4};    // c  = b * 0.5
                8:  uop = {4'd3, 4'd11, 4'd4, 3'd4};    // d  = 1.5 - c
                9:  uop = {4'd1, 4'd3, 4'd4, 3'd5};     // y1 = y0 * d
                default: uop = {4'd9, 11'd0};
            endcase
            default: uop = {4'd9, 11'd0};               // NONE / ABS: nothing to run
        endcase
    end
    wire [3:0] u_unit = uop[14:11], u_a = uop[10:7], u_b = uop[6:3];
    wire [2:0] u_dst = uop[2:0];

    // operand value of lane l
    function [31:0] operand(input [3:0] code, input integer ln);
        reg [31:0] x;
        reg        odd;
        begin
            x = RF[ln][0];
            odd = ~x[23];                                   // exponent field even <=> unbiased exponent odd
            case (code)
                4'd8:  operand = C_LOG2E;
                4'd9:  operand = C_64;
                4'd10: operand = C_2;
                4'd11: operand = C_1_5;
                4'd12: operand = C_0_5;
                4'd13: operand = TT[ln];
                4'd14: operand = SS[ln];
                4'd15: operand = u_unit == 4'd4 ? {15'd0, x[16:0]}                           // frac & 0x1FFFF
                               : func == F_RECIP ? {1'b0, 8'd127, x[22:0]}                    // m
                               : {1'b0, 8'd127 + {7'd0, odd}, x[22:0]};                        // m' (RSQRT)
                default: operand = RF[ln][code[2:0]];
            endcase
        end
    endfunction
    function [3:0] unit_lat(input [3:0] u);
        unit_lat = u == 4'd1 || u == 4'd2 || u == 4'd3 ? 4'd4 : u == 4'd4 || u == 4'd5 || u == 4'd6 ? 4'd2 : 4'd0;
    endfunction

    // function result of lane l (END): ldexp and the special cases, as llm/sfu.py
    function [31:0] fresult(input integer ln);
        reg [31:0] x, y;
        reg [7:0]  e;
        reg        odd;
        reg signed [31:0] n;
        begin
            x = RF[ln][0]; e = x[30:23];
            case (func)
                F_EXP: begin
                    n = RF[ln][2];
                    y = ldexp(RF[ln][7], n < -300 ? -12'sd300 : n > 300 ? 12'sd300 : n[11:0]);
                    if (is_nan(x)) fresult = QNAN;
                    else if (!lt_f(x, 32'h42B20000)) fresult = PINF;               // x >= 89
                    else if (!lt_f(32'hC2D00000, x)) fresult = 32'd0;              // x <= -104
                    else fresult = y;
                end
                F_RECIP: begin
                    y = ldexp(RF[ln][5], 12'sd127 - $signed({4'd0, e})) | {x[31], 31'd0};
                    if (is_nan(x)) fresult = QNAN;
                    else if (e == 8'hFF) fresult = {x[31], 31'd0};
                    else if (e == 0) fresult = {x[31], 8'hFF, 23'd0};
                    else fresult = y;
                end
                default: begin                                                      // RSQRT
                    odd = ~e[0];
                    y = ldexp(RF[ln][5], -((($signed({4'd0, e}) - 12'sd127) - $signed({11'd0, odd})) >>> 1));
                    if (is_nan(x)) fresult = QNAN;
                    else if (e == 0) fresult = PINF;
                    else if (x[31]) fresult = QNAN;
                    else if (e == 8'hFF) fresult = 32'd0;
                    else fresult = y;
                end
            endcase
        end
    endfunction

    wire red_sum = red == 2'd1;
    reg  [31:0] tree_a [0:FL-1], tree_b [0:FL-1];     // FL >= D / 2: one pass per level
    always @* for (l = 0; l < FL; l = l + 1) begin
        tree_a[l] = 2 * l < D ? ACCR[2 * l] : 32'd0;
        tree_b[l] = 2 * l + 1 < D ? ACCR[2 * l + 1] : 32'd0;
    end

    reg retire_seq;                            // a sequenced group finished (with or without a write)
    always @(posedge clk) begin
        sq_m_v <= 0; sq_a_v <= 0; sq_i_v <= 0; sq_f_v <= 0;
        retire_seq <= 0;
        if (!resetn || !active) begin
            sq_st <= Q_IDLE; sq_act <= 0;
        end else case (sq_st)
            Q_IDLE:
                if (seqmode && pv[PL-1]) begin     // A2 result of a half; start after the last one
                    for (l = 0; l < FL; l = l + 1)
                        SQIN[ph[PL-1] * FL + l] <= func == F_ABS ? {1'b0, a2o[l][30:0]} : a2o[l];
                    sq_mask[ph[PL-1] * FL +: FL] <= pmask[PL-1];
                    sq_first <= pfirst[PL-1];
                    sq_last  <= plast[PL-1];
                    sq_dst   <= pdst[PL-1];
                    sq_h     <= 0;
                    if (phl[PL-1]) begin
                        sq_act <= 1;
                        sq_st  <= Q_LOAD;
                    end
                end
            Q_LOAD: begin                          // RF[.][0] = the half's input
                for (l = 0; l < FL; l = l + 1) RF[l][0] <= SQIN[sq_h * FL + l];
                sq_step <= 0;
                sq_st   <= Q_ISSUE;
            end
            Q_ISSUE: begin
                case (u_unit)
                    4'd1: for (l = 0; l < FL; l = l + 1) begin sq_ma[l] <= operand(u_a, l); sq_mb[l] <= operand(u_b, l); end
                    4'd2, 4'd3: begin
                        for (l = 0; l < FL; l = l + 1) begin sq_aa[l] <= operand(u_a, l); sq_ab[l] <= operand(u_b, l); end
                        sq_a_sub <= u_unit == 4'd3;
                    end
                    4'd4: for (l = 0; l < FL; l = l + 1) sq_ix[l] <= operand(u_a, l);
                    4'd5, 4'd6: begin
                        for (l = 0; l < FL; l = l + 1) sq_fx[l] <= operand(u_a, l);
                        sq_f_floor <= 1;
                        sq_f_lo <= u_unit == 4'd6 ? 32'd0 : 32'h8000_0000;
                        sq_f_hi <= u_unit == 4'd6 ? 32'd63 : 32'h7FFF_FFFF;
                    end
                    default: ;
                endcase
                if (u_unit == 4'd9) begin                              // END (or nothing to run)
                    for (l = 0; l < FL; l = l + 1)
                        VV[sq_h * FL + l] <= func == F_NONE || func == F_ABS ? RF[l][0] : fresult(l);
                    if (NH > 1 && sq_h == 0) begin
                        sq_h  <= 1;
                        sq_st <= Q_LOAD;
                    end else
                        sq_st <= Q_POST;
                end else if (u_unit == 4'd7) begin
                    sq_lane <= 0;
                    sq_st   <= Q_LOOK;
                end else if (u_unit == 4'd8) begin
                    for (l = 0; l < FL; l = l + 1) RF[l][u_dst] <= ldexp(RF[l][u_a[2:0]], -12'sd17);
                    sq_step <= sq_step + 1;
                end else begin
                    sq_wait <= unit_lat(u_unit) + 4'd1;                // operands registered this cycle
                    sq_st   <= Q_WAIT;
                end
            end
            Q_WAIT:
                if (sq_wait == 1) begin
                    for (l = 0; l < FL; l = l + 1)
                        RF[l][u_dst] <= u_unit == 4'd1 ? m2o[l] : u_unit == 4'd4 ? ia_y_arr[l]
                                      : u_unit == 4'd5 || u_unit == 4'd6 ? f2o[l] : a2o[l];
                    sq_step <= sq_step + 1;
                    sq_st   <= Q_ISSUE;
                end else
                    sq_wait <= sq_wait - 1;
            Q_LOOK: begin                                              // table rows, one lane per cycle
                begin : lk
                    reg [8:0]  addr;
                    reg [63:0] row_;
                    reg [31:0] xx;
                    xx = RF[sq_lane][0];
                    addr = func == F_EXP   ? {2'd0, 1'b0, RF[sq_lane][5][5:0]}
                         : func == F_RECIP ? {2'd1, 1'b0, xx[22:17]}
                         :                   {2'd2, ~xx[23], xx[22:17]};
                    row_ = sfu_rom(addr);
                    TT[sq_lane] <= row_[63:32];
                    SS[sq_lane] <= row_[31:0];
                end
                if (sq_lane == FL - 1) begin
                    sq_step <= sq_step + 1;
                    sq_st   <= Q_ISSUE;
                end else
                    sq_lane <= sq_lane + 1;
            end
            Q_POST: begin                                              // RELU + VALID
                for (l = 0; l < D; l = l + 1)
                    VV[l] <= sq_mask[l] ? relu_f(VV[l], relu) : red == 2'd0 ? 32'd0 : red_sum ? 32'd0 : NINF;
                sq_h  <= 0;
                sq_st <= red != 0 ? Q_RED : Q_OUT;
            end
            Q_RED:
                if (sq_first) begin
                    for (l = 0; l < D; l = l + 1) ACCR[l] <= VV[l];
                    sq_st <= sq_last ? Q_TREE : Q_FIN;
                    sq_lvl <= 0;
                end else if (red_sum) begin
                    for (l = 0; l < FL; l = l + 1) begin sq_aa[l] <= ACCR[sq_h * FL + l]; sq_ab[l] <= VV[sq_h * FL + l]; end
                    sq_a_sub <= 0;
                    sq_wait  <= 4'd5;
                    sq_st    <= Q_REDW;
                end else begin
                    for (l = 0; l < D; l = l + 1) ACCR[l] <= fmax(ACCR[l], VV[l]);
                    sq_st <= sq_last ? Q_TREE : Q_FIN;
                    sq_lvl <= 0;
                end
            Q_REDW:
                if (sq_wait == 1) begin
                    for (l = 0; l < FL; l = l + 1) ACCR[sq_h * FL + l] <= a2o[l];
                    if (NH > 1 && sq_h == 0) begin
                        sq_h  <= 1;
                        sq_st <= Q_RED;
                    end else
                        sq_st <= sq_last ? Q_TREE : Q_FIN;
                    sq_lvl <= 0;
                end else
                    sq_wait <= sq_wait - 1;
            Q_TREE:                                                    // (l0 op l1), (l2 op l3), ...
                if ((D >> sq_lvl) == 1) begin
                    for (l = 0; l < D; l = l + 1) VV[l] <= ACCR[0];
                    sq_st <= Q_OUT;
                end else if (red_sum) begin
                    for (l = 0; l < FL; l = l + 1) begin sq_aa[l] <= tree_a[l]; sq_ab[l] <= tree_b[l]; end
                    sq_a_sub <= 0;
                    sq_wait  <= 4'd5;
                    sq_st    <= Q_TREEW;
                end else begin
                    for (l = 0; l < FL; l = l + 1) if (2 * l + 1 < D) ACCR[l] <= fmax(tree_a[l], tree_b[l]);
                    sq_lvl <= sq_lvl + 1;
                end
            Q_TREEW:
                if (sq_wait == 1) begin
                    for (l = 0; l < FL; l = l + 1) if (2 * l + 1 < D) ACCR[l] <= a2o[l];
                    sq_lvl <= sq_lvl + 1;
                    sq_st  <= Q_TREE;
                end else
                    sq_wait <= sq_wait - 1;
            Q_OUT:                                                     // output conversion (f2i) if integer
                if (ot == T_F32 || red != 0) begin
                    sq_st <= Q_FIN;
                end else begin
                    for (l = 0; l < FL; l = l + 1) sq_fx[l] <= VV[sq_h * FL + l];
                    sq_f_floor <= 0;
                    sq_f_lo <= ot == T_I8 ? -32'sd127 : 32'h8000_0000;
                    sq_f_hi <= ot == T_I8 ? 32'sd127 : 32'h7FFF_FFFF;
                    sq_wait <= 4'd3;
                    sq_st   <= Q_OUTW;
                end
            Q_OUTW:
                if (sq_wait == 1) begin
                    for (l = 0; l < FL; l = l + 1) VV[sq_h * FL + l] <= f2o[l];
                    if (NH > 1 && sq_h == 0) begin
                        sq_h  <= 1;
                        sq_st <= Q_OUT;
                    end else
                        sq_st <= Q_FIN;
                end else
                    sq_wait <= sq_wait - 1;
            default: begin                                             // Q_FIN: write (unless a row is still open)
                sq_act     <= 0;
                retire_seq <= 1;
                sq_st      <= Q_IDLE;
            end
        endcase
    end
    wire seq_write = sq_st == Q_FIN && (red == 0 || sq_last);
    reg [32*D-1:0] vv_w;
    always @* for (l = 0; l < D; l = l + 1) vv_w[32*l +: 32] = VV[l];

    // ---------------------------------------------------- TRANSPOSE
    reg  [15:0] tk, tj, tkb, tks;               // block, word in block, block row / column
    reg  [1:0]  tph;                            // 0 read, 1 wait last, 2 write
    reg  [32*D-1:0] tbuf [0:D-1];
    reg         tret_v;
    reg  [15:0] tret_j;
    wire        tsp = it == T_I8;
    assign tr_rd   = active && xpose && tph == 0 && tk < (G >> LOGD) && !wr_now && !tw_en;
    assign tr_mem  = m1;
    assign tr_word = b1 + tkb * D * S + tks + tj * S;
    integer ti;

    // ---------------------------------------------------- main control
    integer retire;
    always @(posedge clk) begin
        done <= 0;
        push = 0;
        p0_v <= 0;
        tw_en <= 0;
        if (!resetn) begin
            active <= 0; ret_v <= 0; oq_wp <= 0; oq_rp <= 0; inflight <= 0; tret_v <= 0;
            rd_gap <= 0; p0_more <= 0;
        end else begin
            if (cmd_valid && cmd_ready) begin
                active  <= 1;
                xpose   <= cmd_op[2:0] == VOP_TRANSPOSE;
                m1 <= cmd_src1[31:28]; b1 <= cmd_src1[15:0];
                m2 <= cmd_src2[31:28]; b2 <= cmd_src2[15:0];
                md <= cmd_dst[31:28];  bd <= cmd_dst[15:0];
                G  <= cmd_groups;
                op <= cmd_op[2:0]; relu <= cmd_op[4]; unary <= cmd_op[2:0] == VOP_COPY;
                it <= cmd_types[1:0]; ot <= cmd_types[3:2];
                it2 <= cmd_types[5:4] == 2'd0 ? cmd_types[1:0] : cmd_types[5:4] == 2'd1 ? T_I8 :
                       cmd_types[5:4] == 2'd2 ? T_I32 : T_F32;
                P2 <= cmd_mod; P1 <= cmd_p1; S <= cmd_s;
                func <= cmd_flags[3:1]; mo1 <= cmd_flags[5:4]; mo2 <= cmd_flags[7:6];
                red <= cmd_flags[9:8]; swapneg <= cmd_flags[10];
                seqmode <= cmd_flags[9:8] != 0 || cmd_flags[3:1] == F_EXP || cmd_flags[3:1] == F_RECIP ||
                           cmd_flags[3:1] == F_RSQRT;
                imm <= cmd_imm; cA <= cmd_a; cB <= cmd_b;
                rowlen <= cmd_rowlen == 0 ? cmd_groups : cmd_rowlen;
                vld <= cmd_vld;
                rg <= 0; i1c <= 0; i1d <= 0; i2c <= 0; i2d <= 0; gr <= 0; row <= 0; rpart <= 0;
                tk <= 0; tj <= 0; tkb <= 0; tks <= 0; tph <= 0; tret_v <= 0;
                inflight <= 0;
            end

            // ---- read issue / return (elementwise commands)
            ret_v <= rd_issue;
            if (rd_issue && rd_last) rd_gap <= NH - 1;
            else if (rd_gap != 0) rd_gap <= rd_gap - 1;
            if (rd_issue) begin
                ret_src2 <= rpart;
                ret_mem  <= rd_mem;
                ret_last <= rd_last;
                if (!rpart) begin
                    ret_mask    <= g_mask;
                    ret_first   <= gr == 0;
                    ret_rowlast <= gr == rowlen - 1;
                    ret_dst     <= red != 0 ? bd + row : bd + rg;
                end
                if (rd_last) begin
                    rpart <= 0;
                    rg    <= rg + 1;
                    if (i1c + 1 == P1) begin i1c <= 0; i1d <= i1d + 1; end else i1c <= i1c + 1;
                    if (i2c + 1 == P2) begin i2c <= 0; i2d <= i2d + 1; end else i2c <= i2c + 1;
                    if (gr + 1 == rowlen) begin gr <= 0; row <= row + 1; end else gr <= gr + 1;
                end else
                    rpart <= 1;
            end
            if (p0_more) begin                                  // the next half of the group
                p0_v    <= 1;
                p0_h    <= 1;
                p0_more <= 0;
            end
            if (ret_v) begin
                if (!ret_src2) rawa <= ret_data; else rawb <= ret_data;
                if (ret_last) begin
                    p0_v     <= 1;
                    p0_h     <= 0;
                    p0_more  <= NH > 1;
                    p0_gmask <= ret_mask;
                    p0_first <= ret_first;
                    p0_last  <= ret_rowlast;
                    p0_dst   <= ret_dst;
                end
            end

            // ---- results -> output FIFO
            if (sp_v[2] && sp_hl[2]) begin
                push = 1;
                push_d = sp_word;
                push_a = sp_dst[2];
            end
            if (seq_write) begin
                push = 1;
                push_d = vv_w;
                push_a = sq_dst;
            end
            if (push) begin
                oq_d[oq_wp[4:0]] <= push_d;
                oq_a[oq_wp[4:0]] <= push_a;
                oq_wp <= oq_wp + 1;
            end
            if (wr_now && !tw_en) oq_rp <= oq_rp + 1;
            retire = (wr_now && !tw_en ? 1 : 0) + (retire_seq && !(red == 0 || sq_last) ? 1 : 0);
            inflight <= inflight + (rd_issue && !rpart ? 1 : 0) - retire;

            // ---- TRANSPOSE: read D words of a block, then write D words
            tret_v <= tr_rd;
            tret_j <= tj;
            if (tr_rd) begin
                if (tj == D - 1) begin tj <= 0; tph <= 1; end else tj <= tj + 1;
            end
            if (tret_v) tbuf[tret_j] <= ret_mem_x(tsp);
            if (active && xpose && tph == 1 && !tret_v) begin tph <= 2; tj <= 0; end
            if (active && xpose && tph == 2) begin
                tw_en   <= 1;
                tw_addr <= bd + tk * D + tj;
                for (ti = 0; ti < D; ti = ti + 1)
                    if (tsp) tw_d[8*ti +: 8] <= tbuf[ti][8*tj[LOGD-1:0] +: 8];
                    else     tw_d[32*ti +: 32] <= tbuf[ti][32*tj[LOGD-1:0] +: 32];
                if (tj == D - 1) begin
                    tph <= 0; tj <= 0; tk <= tk + 1;
                    if (tks + 1 == S) begin tks <= 0; tkb <= tkb + 1; end else tks <= tks + 1;
                end else
                    tj <= tj + 1;
            end

            // ---- completion
            if (active && !xpose && rg == G && inflight == 0 && oq_empty && sq_st == Q_IDLE && !(cmd_valid && cmd_ready)) begin
                active <= 0;
                done   <= 1;
            end
            if (active && xpose && tk == (G >> LOGD) && tph == 0 && !tw_en && !(cmd_valid && cmd_ready)) begin
                active <= 0;
                done   <= 1;
            end
        end
    end
    function [32*D-1:0] ret_mem_x(input sp);
        ret_mem_x = sp ? {{24*D{1'b0}}, m1 == MEM_SPAD_B ? sb_dout : sa_dout} : ac_dout;
    endfunction

    assign perf_ev = {wr_now || tw_en, sq_act, rd_can && rd_block, active};
endmodule
