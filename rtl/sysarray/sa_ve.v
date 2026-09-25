// VE engine: vector operations on the local memories (docs/double_buffer_design.md §6, M3).
//
// A command processes `groups` groups of VL = D elements:
//     dst[g] = post(op(src1[g], src2[g']))      g' = g, 0 (broadcast) or g mod period
// op   : ADD, SUB (int32 saturating), MUL (int8/int16 inputs), MAX, MIN, COPY (unary)
// post : optional RELU, optional REQUANT  y = ((x*scale + 2^(shift-1)) >> shift) + zp,
//        then clamp to [lo, hi] and saturate to the output type.
// Types live in fixed memories: int32 in ACC (one word per group), int8 in SPAD
// (one word), int16 in SPAD (two words: elements 0..D/2-1, then D/2..D-1).
//
// Pipeline: a read unit issues one read per cycle (src1 words, then src2 words;
// a broadcast src2 is read once); a group whose operands are complete runs
// through four compute stages into an output FIFO; the write stage writes one
// word per cycle and has priority over reads on the same memory port. At most
// eight groups (the FIFO depth) are in flight. One command at a time; `done` after its last write.
`timescale 1ns / 1ps
`include "sa_macros.vh"

module sa_ve #(
    parameter integer D       = 8,
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
    input  wire [15:0]          cmd_mod,
    input  wire [15:0]          cmd_scale,
    input  wire [4:0]           cmd_shift,
    input  wire [31:0]          cmd_zp,
    input  wire [31:0]          cmd_lo,
    input  wire [31:0]          cmd_hi,
    output reg                  done,
    output wire                 busy,

    // one port per memory (SPAD_A / SPAD_B side B, ACC side A)
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

    // performance events: {group written, out of credits (FIFO full),
    //                      read lost the port to a write, command active}
    output wire [3:0]           perf_ev
);
    `include "sa_defs.vh"

    // ------------------------------------------------------------ command
    reg         active;
    reg  [3:0]  m1, m2, md;               // memories of src1 / src2 / dst
    reg  [15:0] b1, b2, bd;               // base words
    reg  [15:0] ngroups;
    reg  [2:0]  op;
    reg         relu, requant, unary;
    reg  [1:0]  it, ot;
    reg  [15:0] period;
    reg  signed [15:0] scale;
    reg  [4:0]  shift;
    reg signed [47:0] rnd_c;               // REQUANT rounding constant 2^(shift-1)
    reg  signed [31:0] zp, lo, hi;

    wire [1:0] wg_in  = it == VT_I16 ? 2'd2 : 2'd1;   // words per group
    wire [1:0] wg_out = ot == VT_I16 ? 2'd2 : 2'd1;

    assign cmd_ready = !active && resetn;
    assign busy      = active;

    // ----------------------------------------------------- write stage
    // output FIFO of groups: {word1, word0, nwords, dst word}
    localparam integer OQ = 8;
    reg  [32*D-1:0] oq_w0 [0:OQ-1];
    reg  [32*D-1:0] oq_w1 [0:OQ-1];
    reg  [15:0]     oq_addr [0:OQ-1];
    reg  [3:0]      oq_wp, oq_rp;
    reg             wpart;                 // word of the head group being written
    wire            oq_empty = oq_wp == oq_rp;
    wire            wr_now   = !oq_empty;
    wire [15:0]     wr_word  = oq_addr[oq_rp[2:0]] + wpart;
    wire [32*D-1:0] wr_data  = wpart ? oq_w1[oq_rp[2:0]] : oq_w0[oq_rp[2:0]];
    wire            wr_last  = wpart == wg_out - 1;

    // ------------------------------------------------------- read unit
    reg  [15:0] rg;                        // group being read
    reg  [15:0] rg2;                       // its src2 group (g, 0 or g mod period)
    reg  [2:0]  rpart;                     // 0..wg_in-1 src1, then src2
    reg         reading;                   // a group's reads are in progress
    reg  [3:0]  inflight;                  // groups started, not yet written
    reg         b_req;                     // broadcast src2 already requested (read once)
    reg  [2:0]  g_nreads;                  // reads of the group being read (latched at its start), up to 4
    reg  [15:0] groups_written;

    wire        src2_needed = !unary && !(period == 16'd1 && b_req);
    wire [2:0]  start_reads = wg_in + (src2_needed ? wg_in : 2'd0);
    wire [2:0]  nreads      = reading ? g_nreads : start_reads;
    wire        rd_src2     = rpart >= wg_in;
    wire [1:0]  rd_sub      = rd_src2 ? rpart - wg_in : rpart[1:0];
    wire [3:0]  rd_mem      = rd_src2 ? m2 : m1;
    wire [15:0] rd_word     = rd_src2 ? b2 + rg2 * wg_in + rd_sub : b1 + rg * wg_in + rd_sub;
    wire        rd_can      = active && (reading || (rg < ngroups && inflight < OQ));
    wire        rd_block    = wr_now && md == rd_mem;          // write has the port
    wire        rd_issue    = rd_can && !rd_block;

    // --------------------------------------------------- memory ports
    always @* begin
        sa_en = 0; sa_we = 0; sa_addr = 0; sa_din = 0;
        sb_en = 0; sb_we = 0; sb_addr = 0; sb_din = 0;
        ac_en = 0; ac_we = 0; ac_addr = 0; ac_din = 0;
        if (wr_now)
            case (md)
                MEM_SPAD_A: begin sa_en = 1; sa_we = {D{1'b1}};   sa_addr = wr_word; sa_din = wr_data[8*D-1:0]; end
                MEM_SPAD_B: begin sb_en = 1; sb_we = {D{1'b1}};   sb_addr = wr_word; sb_din = wr_data[8*D-1:0]; end
                default:    begin ac_en = 1; ac_we = {4*D{1'b1}}; ac_addr = wr_word; ac_din = wr_data;          end
            endcase
        if (rd_issue)
            case (rd_mem)
                MEM_SPAD_A: begin sa_en = 1; sa_addr = rd_word; end
                MEM_SPAD_B: begin sb_en = 1; sb_addr = rd_word; end
                default:    begin ac_en = 1; ac_addr = rd_word; end
            endcase
    end

    // read return (one cycle later)
    reg         ret_v, ret_src2, ret_last;
    reg  [1:0]  ret_sub;
    reg  [3:0]  ret_mem;
    wire [32*D-1:0] ret_data = ret_mem == MEM_ACC    ? ac_dout :
                               ret_mem == MEM_SPAD_B ? {{24*D{1'b0}}, sb_dout} :
                                                       {{24*D{1'b0}}, sa_dout};

    // operand collection
    reg  [32*D-1:0] opa0, opa1, opb0, opb1;
    reg             grp_ready;              // operands of one group complete (1 cycle)

    // ------------------------------------------------ compute pipeline
    function signed [31:0] elem(input [32*D-1:0] w0, input [32*D-1:0] w1, input [1:0] t, input integer e);
        begin
            if (t == VT_I8)       elem = $signed(w0[8*e +: 8]);
            else if (t == VT_I16) elem = e < D/2 ? $signed(w0[16*e +: 16]) : $signed(w1[16*(e - D/2) +: 16]);
            else                  elem = $signed(w0[32*e +: 32]);
        end
    endfunction

    function signed [31:0] sat32(input signed [32:0] x);
        sat32 = x > 33'sd2147483647 ? 32'sh7FFFFFFF : x < -33'sd2147483648 ? 32'sh80000000 : x[31:0];
    endfunction

    reg              s1_v, s2_v, s3_v, s4_v;
    reg signed [31:0] s1_r [0:D-1];
    reg signed [47:0] s2_p [0:D-1];
    reg signed [47:0] s3_q [0:D-1];
    reg [32*D-1:0]   s4_w0, s4_w1;

    integer e;
    reg signed [31:0] xa, xb;
    reg signed [32:0] wide;
    reg signed [31:0] rv;
    reg signed [48:0] q;
    reg signed [31:0] c;
    reg signed [31:0] tmin, tmax;
    always @(posedge clk) begin
        // stage 1: element op
        s1_v <= grp_ready;
        if (grp_ready)
            for (e = 0; e < D; e = e + 1) begin
                xa = elem(opa0, opa1, it, e);
                xb = elem(opb0, opb1, it, e);
                case (op)
                    VOP_ADD: s1_r[e] <= sat32({xa[31], xa} + {xb[31], xb});
                    VOP_SUB: s1_r[e] <= sat32({xa[31], xa} - {xb[31], xb});
                    VOP_MUL: s1_r[e] <= $signed(xa[15:0]) * $signed(xb[15:0]);
                    VOP_MAX: s1_r[e] <= xa > xb ? xa : xb;
                    VOP_MIN: s1_r[e] <= xa < xb ? xa : xb;
                    default: s1_r[e] <= xa;                     // COPY
                endcase
            end
        // stage 2: RELU, REQUANT multiply
        s2_v <= s1_v;
        if (s1_v)
            for (e = 0; e < D; e = e + 1) begin
                rv = relu && s1_r[e] < 0 ? 32'sd0 : s1_r[e];
                // separate branches: a ?: with an (unsigned) concatenation would
                // make the multiply unsigned and break negative scales
                if (requant) s2_p[e] <= rv * scale;
                else         s2_p[e] <= rv;                  // sign-extended (both signed)
            end
        // stage 3: rounding shift (REQUANT)
        s3_v <= s2_v;
        if (s2_v)
            for (e = 0; e < D; e = e + 1)
                if (requant) s3_q[e] <= (s2_p[e] + rnd_c) >>> shift;
                else         s3_q[e] <= s2_p[e];
        // stage 4: zero point, clamp, saturate to the output type, pack
        s4_v <= s3_v;
        if (s3_v) begin
            tmin = ot == VT_I8 ? -32'sd128   : ot == VT_I16 ? -32'sd32768 : 32'sh80000000;
            tmax = ot == VT_I8 ?  32'sd127   : ot == VT_I16 ?  32'sd32767 : 32'sh7FFFFFFF;
            for (e = 0; e < D; e = e + 1) begin
                q = requant ? s3_q[e] + zp : s3_q[e];
                c = q < lo ? lo : q > hi ? hi : q[31:0];
                c = c < tmin ? tmin : c > tmax ? tmax : c;
                if (ot == VT_I8)
                    s4_w0[8*e +: 8] <= c[7:0];
                else if (ot == VT_I16) begin
                    if (e < D/2) s4_w0[16*e +: 16] <= c[15:0];
                    else         s4_w1[16*(e - D/2) +: 16] <= c[15:0];
                end else
                    s4_w0[32*e +: 32] <= c;
            end
        end
    end

    // ------------------------------------------------------ control
    reg [15:0] wg;                         // group index whose result enters the FIFO next
    always @(posedge clk) begin
        done      <= 0;
        grp_ready <= 0;
        if (!resetn) begin
            active <= 0; reading <= 0; ret_v <= 0;
            oq_wp <= 0; oq_rp <= 0; wpart <= 0; inflight <= 0;
        end else begin
            // accept
            if (cmd_valid && cmd_ready) begin
                active  <= 1;
                m1 <= cmd_src1[31:28]; b1 <= cmd_src1[15:0];
                m2 <= cmd_src2[31:28]; b2 <= cmd_src2[15:0];
                md <= cmd_dst[31:28];  bd <= cmd_dst[15:0];
                ngroups <= cmd_groups;
                op      <= cmd_op[2:0];
                relu    <= cmd_op[4];
                requant <= cmd_op[5];
                unary   <= cmd_op[2:0] == VOP_COPY;
                it      <= cmd_types[1:0];
                ot      <= cmd_types[3:2];
                period  <= cmd_mod;
                scale   <= cmd_scale;
                shift   <= cmd_shift;
                rnd_c   <= cmd_shift == 0 ? 48'sd0 : 48'sd1 <<< (cmd_shift - 1);
                zp      <= cmd_zp;
                lo      <= cmd_lo;
                hi      <= cmd_hi;
                rg <= 0; rg2 <= 0; rpart <= 0; b_req <= 0;
                groups_written <= 0; wg <= 0;
            end

            // read unit
            inflight <= inflight + (rd_issue && !reading) - (wr_now && wr_last);
            ret_v <= rd_issue;
            if (rd_issue) begin
                ret_src2 <= rd_src2;
                ret_sub  <= rd_sub;
                ret_mem  <= rd_mem;
                ret_last <= rpart == nreads - 1;
                if (!reading) g_nreads <= start_reads;
                if (rd_src2 && period == 16'd1) b_req <= 1;
                reading  <= 1;
                if (rpart == nreads - 1) begin
                    reading <= 0;
                    rpart   <= 0;
                    rg      <= rg + 1;
                    rg2     <= period == 0 ? rg + 1 : (rg2 + 1 == period ? 16'd0 : rg2 + 1);
                end else
                    rpart <= rpart + 1;
            end

            if (ret_v) begin
                if (!ret_src2) begin
                    if (ret_sub == 0) opa0 <= ret_data; else opa1 <= ret_data;
                end else begin
                    if (ret_sub == 0) opb0 <= ret_data; else opb1 <= ret_data;
                end
                if (ret_last) grp_ready <= 1;
            end

            // result -> output FIFO
            if (s4_v) begin
                oq_w0[oq_wp[2:0]]   <= s4_w0;
                oq_w1[oq_wp[2:0]]   <= s4_w1;
                oq_addr[oq_wp[2:0]] <= bd + wg * wg_out;
                oq_wp               <= oq_wp + 1;
                wg                  <= wg + 1;
            end

            // write stage
            if (wr_now) begin
                if (wr_last) begin
                    wpart  <= 0;
                    oq_rp  <= oq_rp + 1;
                    groups_written <= groups_written + 1;
                    if (groups_written + 1 == ngroups) begin
                        active <= 0;
                        done   <= 1;
                    end
                end else
                    wpart <= 1;
            end
        end
    end

    assign perf_ev = {wr_now && wr_last, active && !reading && rg < ngroups && inflight == OQ,
                      rd_can && rd_block, active};
endmodule
