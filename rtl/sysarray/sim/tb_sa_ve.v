// Unit test for sa_ve against a behavioral reference of the vector semantics
// (op, RELU, REQUANT with rounding / zero point / clamp, saturation to the
// output type, src2 period / broadcast, int8 / int16 / int32 layouts).
// Directed cases (fused GEMM epilogue, in-place op, saturation) and a random
// stream of commands; after each command the memories are compared with the
// reference model.
`timescale 1ns / 1ps
`include "sa_macros.vh"

module tb_sa_ve;
    `include "sa_defs.vh"
    localparam integer D  = 8;
    localparam integer SW = 16384, CW = 8192;
    localparam integer SAW = 14, CAW = 13;

    reg clk = 0, resetn = 0;
    always #10 clk = ~clk;
    integer errors = 0, seed = 31337;
    function integer rnd(input integer max); rnd = $unsigned($random(seed)) % (max + 1); endfunction

    // ------------------------------------------------------------------ DUT
    reg         cmd_valid = 0;
    wire        cmd_ready, done, busy;
    reg  [31:0] c_src1, c_src2, c_dst, c_zp, c_lo, c_hi;
    reg  [15:0] c_groups, c_mod, c_scale;
    reg  [7:0]  c_op;
    reg  [5:0]  c_types;
    reg  [4:0]  c_shift;
    wire        sa_en, sb_en, ac_en;
    wire [D-1:0] sa_we, sb_we;
    wire [4*D-1:0] ac_we;
    wire [SAW-1:0] sa_addr, sb_addr;
    wire [CAW-1:0] ac_addr;
    wire [8*D-1:0] sa_din, sb_din, sa_dout, sb_dout;
    wire [32*D-1:0] ac_din, ac_dout;

    sa_ve #(.D(D), .SPAD_AW(SAW), .ACC_AW(CAW)) dut (
        .clk(clk), .resetn(resetn), .cmd_valid(cmd_valid), .cmd_ready(cmd_ready),
        .cmd_src1(c_src1), .cmd_src2(c_src2), .cmd_dst(c_dst), .cmd_groups(c_groups), .cmd_op(c_op),
        .cmd_types(c_types), .cmd_mod(c_mod), .cmd_scale(c_scale), .cmd_shift(c_shift), .cmd_zp(c_zp),
        .cmd_lo(c_lo), .cmd_hi(c_hi), .done(done), .busy(busy),
        .sa_en(sa_en), .sa_we(sa_we), .sa_addr(sa_addr), .sa_din(sa_din), .sa_dout(sa_dout),
        .sb_en(sb_en), .sb_we(sb_we), .sb_addr(sb_addr), .sb_din(sb_din), .sb_dout(sb_dout),
        .ac_en(ac_en), .ac_we(ac_we), .ac_addr(ac_addr), .ac_din(ac_din), .ac_dout(ac_dout));

    wire [8*D-1:0] nu_a, nu_b; wire [32*D-1:0] nu_c;
    sa_bankmem #(.W(8*D), .DEPTH(SW)) spad_a (.clk(clk),
        .a_en(1'b0), .a_we({D{1'b0}}), .a_addr({SAW{1'b0}}), .a_din({8*D{1'b0}}), .a_dout(nu_a),
        .b_en(sa_en), .b_we(sa_we), .b_addr(sa_addr), .b_din(sa_din), .b_dout(sa_dout));
    sa_bankmem #(.W(8*D), .DEPTH(SW)) spad_b (.clk(clk),
        .a_en(1'b0), .a_we({D{1'b0}}), .a_addr({SAW{1'b0}}), .a_din({8*D{1'b0}}), .a_dout(nu_b),
        .b_en(sb_en), .b_we(sb_we), .b_addr(sb_addr), .b_din(sb_din), .b_dout(sb_dout));
    sa_bankmem #(.W(32*D), .DEPTH(CW)) accm (.clk(clk),
        .a_en(ac_en), .a_we(ac_we), .a_addr(ac_addr), .a_din(ac_din), .a_dout(ac_dout),
        .b_en(1'b0), .b_we({4*D{1'b0}}), .b_addr({CAW{1'b0}}), .b_din({32*D{1'b0}}), .b_dout(nu_c));

    // ------------------------------------------- memory views and shadows
    reg [63:0]  sh_a [0:SW-1];
    reg [63:0]  sh_b [0:SW-1];
    reg [255:0] sh_c [0:CW-1];
    task hw_set(input [3:0] m, input integer w, input [255:0] v);
        case (m)
            MEM_SPAD_A: if (w < SW/2) spad_a.bank[0].ram.ram_block[w] = v[63:0]; else spad_a.bank[1].ram.ram_block[w - SW/2] = v[63:0];
            MEM_SPAD_B: if (w < SW/2) spad_b.bank[0].ram.ram_block[w] = v[63:0]; else spad_b.bank[1].ram.ram_block[w - SW/2] = v[63:0];
            default:    if (w < CW/2) accm.bank[0].ram.ram_block[w]   = v;        else accm.bank[1].ram.ram_block[w - CW/2]   = v;
        endcase
    endtask
    function [255:0] hw_get(input [3:0] m, input integer w);
        case (m)
            MEM_SPAD_A: hw_get = w < SW/2 ? spad_a.bank[0].ram.ram_block[w] : spad_a.bank[1].ram.ram_block[w - SW/2];
            MEM_SPAD_B: hw_get = w < SW/2 ? spad_b.bank[0].ram.ram_block[w] : spad_b.bank[1].ram.ram_block[w - SW/2];
            default:    hw_get = w < CW/2 ? accm.bank[0].ram.ram_block[w]   : accm.bank[1].ram.ram_block[w - CW/2];
        endcase
    endfunction
    function [255:0] sh_get(input [3:0] m, input integer w);
        sh_get = m == MEM_SPAD_A ? {192'd0, sh_a[w]} : m == MEM_SPAD_B ? {192'd0, sh_b[w]} : sh_c[w];
    endfunction
    task sh_set(input [3:0] m, input integer w, input [255:0] v);
        begin
            if (m == MEM_SPAD_A) sh_a[w] = v[63:0];
            else if (m == MEM_SPAD_B) sh_b[w] = v[63:0];
            else sh_c[w] = v;
        end
    endtask
    task fill(input [3:0] m, input integer from, input integer n);
        integer w, j;
        reg [255:0] v;
        for (w = from; w < from + n; w = w + 1) begin
            for (j = 0; j < 8; j = j + 1) v[32*j +: 32] = $random(seed);
            if (m != MEM_ACC) v[255:64] = 0;
            sh_set(m, w, v);
            hw_set(m, w, v);
        end
    endtask

    // ---------------------------------------------------- reference model
    function integer wpg(input [1:0] t); wpg = t == VT_I16 ? 2 : 1; endfunction
    function signed [31:0] ref_elem(input [3:0] m, input integer base, input [1:0] t, input integer e);
        reg [255:0] w0, w1;
        begin
            w0 = sh_get(m, base); w1 = sh_get(m, base + 1);
            if (t == VT_I8)       ref_elem = $signed(w0[8*e +: 8]);
            else if (t == VT_I16) ref_elem = e < D/2 ? $signed(w0[16*e +: 16]) : $signed(w1[16*(e - D/2) +: 16]);
            else                  ref_elem = $signed(w0[32*e +: 32]);
        end
    endfunction
    function signed [31:0] sat(input signed [63:0] x, input signed [63:0] lo_, input signed [63:0] hi_);
        sat = x < lo_ ? lo_ : x > hi_ ? hi_ : x;
    endfunction

    reg signed [31:0] res [0:4095][0:D-1];      // results of one command (applied after all reads)
    task ref_ve;
        integer g, g2, e;
        reg signed [63:0] a, b, r, q, tmin, tmax;
        reg [255:0] w0, w1;
        begin
            tmin = c_types[3:2] == VT_I8 ? -128 : c_types[3:2] == VT_I16 ? -32768 : -64'sd2147483648;
            tmax = c_types[3:2] == VT_I8 ?  127 : c_types[3:2] == VT_I16 ?  32767 :  64'sd2147483647;
            for (g = 0; g < c_groups; g = g + 1) begin
                g2 = c_mod == 0 ? g : c_mod == 1 ? 0 : g % c_mod;
                for (e = 0; e < D; e = e + 1) begin
                    a = ref_elem(c_src1[31:28], c_src1[15:0] + g  * wpg(c_types[1:0]), c_types[1:0], e);
                    b = ref_elem(c_src2[31:28], c_src2[15:0] + g2 * wpg(c_types[1:0]), c_types[1:0], e);
                    case (c_op[2:0])
                        VOP_ADD: r = sat(a + b, -64'sd2147483648, 64'sd2147483647);
                        VOP_SUB: r = sat(a - b, -64'sd2147483648, 64'sd2147483647);
                        VOP_MUL: r = $signed(a[15:0]) * $signed(b[15:0]);
                        VOP_MAX: r = a > b ? a : b;
                        VOP_MIN: r = a < b ? a : b;
                        default: r = a;
                    endcase
                    if (c_op[4] && r < 0) r = 0;
                    if (c_op[5])
                        q = ((r * $signed(c_scale) + (c_shift == 0 ? 0 : (64'sd1 <<< (c_shift - 1)))) >>> c_shift) + $signed(c_zp);
                    else
                        q = r;
                    q = sat(q, $signed(c_lo), $signed(c_hi));
                    res[g][e] = sat(q, tmin, tmax);
                end
            end
            for (g = 0; g < c_groups; g = g + 1) begin
                w0 = 0; w1 = 0;
                for (e = 0; e < D; e = e + 1)
                    if (c_types[3:2] == VT_I8)       w0[8*e +: 8] = res[g][e][7:0];
                    else if (c_types[3:2] == VT_I16) begin
                        if (e < D/2) w0[16*e +: 16] = res[g][e][15:0]; else w1[16*(e - D/2) +: 16] = res[g][e][15:0];
                    end else                         w0[32*e +: 32] = res[g][e];
                sh_set(c_dst[31:28], c_dst[15:0] + g * wpg(c_types[3:2]), w0);
                if (c_types[3:2] == VT_I16) sh_set(c_dst[31:28], c_dst[15:0] + g * 2 + 1, w1);
            end
        end
    endtask

    // ------------------------------------------------------------ driving
    integer ncmd = 0, cyc0, last_cycles;
    reg     pipelined = 0;          // issue the next command as soon as the engine accepts it
    task run(input [3:0] m1, input integer w1, input [3:0] m2, input integer w2, input [3:0] md, input integer wd,
             input integer groups, input [2:0] op, input relu, input requant, input [1:0] it, input [1:0] ot,
             input integer period, input integer scale, input integer shift, input integer zp,
             input integer lo, input integer hi);
        begin
            c_src1 = {m1, 12'd0, w1[15:0]}; c_src2 = {m2, 12'd0, w2[15:0]}; c_dst = {md, 12'd0, wd[15:0]};
            c_groups = groups; c_op = {2'b00, requant, relu, 1'b0, op}; c_types = {ot, it};
            c_mod = period; c_scale = scale; c_shift = shift; c_zp = zp; c_lo = lo; c_hi = hi;
            ref_ve;
            @(posedge clk); #1 cmd_valid = 1;
            cyc0 = $time;
            @(posedge clk); while (!cmd_ready) @(posedge clk);
            #1 cmd_valid = 0;
            if (!pipelined) begin
                @(posedge clk); while (!done) @(posedge clk);
                last_cycles = ($time - cyc0) / 20;
                check_all(md, wd, groups * wpg(ot));
            end
            ncmd = ncmd + 1;
        end
    endtask

    // compare one word with the reference. Function results go through regs:
    // xsim can misreport `!==` between two function calls in one expression
    reg [255:0] cmp_h, cmp_r;
    task cmp(input [3:0] m, input integer w);
        begin
            cmp_h = hw_get(m, w); cmp_r = sh_get(m, w);
            if (cmp_h !== cmp_r) begin
                errors = errors + 1;
                if (errors <= 10) $display("TB ERROR cmd %0d (op %0d it %0d ot %0d relu %0d rq %0d): mem %0d word %0d = %h, expected %h",
                                           ncmd, c_op[2:0], c_types[1:0], c_types[3:2], c_op[4], c_op[5], m, w, cmp_h, cmp_r);
            end
        end
    endtask
    // the destination region (+ neighbours)
    task check_all(input [3:0] m, input integer from, input integer n);
        integer w;
        for (w = from - 2; w < from + n + 2; w = w + 1)
            if (w >= 0 && w < (m == MEM_ACC ? CW : SW)) cmp(m, w);
    endtask

    localparam integer I32MIN = 32'h80000000, I32MAX = 32'h7FFFFFFF;
    integer n, it_, ot_, m1_, m2_, md_, groups, epi_cycles;
    initial begin
        fill(MEM_SPAD_A, 0, SW); fill(MEM_SPAD_B, 0, SW); fill(MEM_ACC, 0, CW);
        repeat (5) @(posedge clk);
        #1 resetn = 1;

        // --- fused GEMM epilogue: C strip (8 x 64 int32, row-major in ACC) + bias
        //     row (64 int32 = 8 words, period 8) -> RELU -> REQUANT -> int8 SPAD
        run(MEM_ACC, 100, MEM_ACC, 4200, MEM_SPAD_A, 300, 64, VOP_ADD, 1, 1, VT_I32, VT_I8,
            8, 181, 15, -3, I32MIN, I32MAX);
        epi_cycles = last_cycles;
        // --- same without RELU, output int32 in place (dst = src1)
        run(MEM_ACC, 100, MEM_ACC, 4200, MEM_ACC, 100, 64, VOP_ADD, 0, 0, VT_I32, VT_I32,
            8, 1, 0, 0, I32MIN, I32MAX);
        // --- int8 elementwise: every op, SPAD_A x SPAD_B -> SPAD_B
        for (n = 0; n <= 5; n = n + 1)
            run(MEM_SPAD_A, 1000, MEM_SPAD_B, 2000, MEM_SPAD_B, 9000 + 64*n, 32, n, 0, 0, VT_I8, VT_I8,
                0, 1, 0, 0, I32MIN, I32MAX);
        // --- int8 -> int16 / int32 widening, int16 -> int16, int16 MUL -> int32 (ACC)
        run(MEM_SPAD_A, 1000, MEM_SPAD_B, 2000, MEM_SPAD_A, 12000, 16, VOP_MUL, 0, 0, VT_I8, VT_I16, 0, 1, 0, 0, I32MIN, I32MAX);
        run(MEM_SPAD_A, 1000, MEM_SPAD_B, 2000, MEM_ACC,    6000,  16, VOP_SUB, 0, 0, VT_I8, VT_I32, 0, 1, 0, 0, I32MIN, I32MAX);
        run(MEM_SPAD_A, 3000, MEM_SPAD_B, 3000, MEM_SPAD_A, 12100, 16, VOP_ADD, 0, 0, VT_I16, VT_I16, 0, 1, 0, 0, I32MIN, I32MAX);
        run(MEM_SPAD_A, 3000, MEM_SPAD_B, 3000, MEM_ACC,    6100,  16, VOP_MUL, 1, 0, VT_I16, VT_I32, 0, 1, 0, 0, I32MIN, I32MAX);
        // --- broadcast src2 (period 1), clamp window, same memory for all operands (port sharing)
        run(MEM_SPAD_A, 4000, MEM_SPAD_A, 4500, MEM_SPAD_A, 4600, 40, VOP_MAX, 0, 0, VT_I8, VT_I8, 1, 1, 0, 0, -20, 30);
        run(MEM_ACC, 200, MEM_ACC, 7000, MEM_ACC, 7100, 24, VOP_ADD, 0, 1, VT_I32, VT_I32, 1, -300, 7, 1000, -500000, 500000);
        // --- saturation: int32 ADD overflow, int8 narrowing
        hw_set(MEM_ACC, 7500, {8{32'h7FFFFFF0}}); sh_set(MEM_ACC, 7500, {8{32'h7FFFFFF0}});
        hw_set(MEM_ACC, 7501, {8{32'h00000100}}); sh_set(MEM_ACC, 7501, {8{32'h00000100}});
        run(MEM_ACC, 7500, MEM_ACC, 7501, MEM_ACC, 7502, 1, VOP_ADD, 0, 0, VT_I32, VT_I32, 0, 1, 0, 0, I32MIN, I32MAX);
        cmp_h = hw_get(MEM_ACC, 7502);
        if (cmp_h !== {8{32'h7FFFFFFF}}) begin errors = errors + 1; $display("TB ERROR: int32 ADD did not saturate"); end
        // --- bank-crossing ranges
        run(MEM_SPAD_A, SW/2 - 10, MEM_SPAD_B, SW/2 - 3, MEM_SPAD_B, SW/2 + 100, 20, VOP_SUB, 1, 1, VT_I8, VT_I8, 3, 77, 3, 5, I32MIN, I32MAX);

        // --- random stream
        for (n = 0; n < 200; n = n + 1) begin
            it_ = rnd(2); ot_ = rnd(2);
            m1_ = it_ == VT_I32 ? MEM_ACC : 1 + rnd(1);
            m2_ = it_ == VT_I32 ? MEM_ACC : 1 + rnd(1);
            md_ = ot_ == VT_I32 ? MEM_ACC : 1 + rnd(1);
            groups = 1 + rnd(20);
            run(m1_, rnd(40) + (it_ == VT_I32 ? 0 : 1000), m2_, 200 + rnd(40) + (it_ == VT_I32 ? 0 : 1000),
                md_, (ot_ == VT_I32 ? 3000 : 6000) + 64 * (n % 32),
                groups, it_ == VT_I32 ? rnd(4) - (rnd(4) == 2 ? 0 : 0) : rnd(5), rnd(1), rnd(1), it_, ot_,
                rnd(3), $signed(rnd(4000)) - 2000, rnd(20), $signed(rnd(200)) - 100,
                rnd(1) ? I32MIN : -rnd(100000), rnd(1) ? I32MAX : rnd(100000));
        end

        // --- back-to-back issue: the next command is accepted right after the
        //     previous one's last write (disjoint regions), then compare everything
        pipelined = 1;
        for (n = 0; n < 60; n = n + 1) begin
            it_ = rnd(2); ot_ = rnd(2);
            m1_ = it_ == VT_I32 ? MEM_ACC : 1 + rnd(1);
            m2_ = it_ == VT_I32 ? MEM_ACC : 1 + rnd(1);
            md_ = ot_ == VT_I32 ? MEM_ACC : 1 + rnd(1);
            groups = 1 + rnd(20);
            run(m1_, rnd(40) + (it_ == VT_I32 ? 0 : 1000), m2_, 200 + rnd(40) + (it_ == VT_I32 ? 0 : 1000),
                md_, (ot_ == VT_I32 ? 3000 : 6000) + 64 * (n % 32),
                groups, rnd(5), rnd(1), rnd(1), it_, ot_,
                rnd(3), $signed(rnd(4000)) - 2000, rnd(20), $signed(rnd(200)) - 100,
                rnd(1) ? I32MIN : -rnd(100000), rnd(1) ? I32MAX : rnd(100000));
        end
        while (busy) @(posedge clk);
        repeat (5) @(posedge clk);
        for (n = 0; n < SW; n = n + 1) begin
            cmp(MEM_SPAD_A, n); cmp(MEM_SPAD_B, n);
            if (n < CW) cmp(MEM_ACC, n);
        end

        $display("TB %s: %0d vector commands, %0d errors; fused epilogue (64 groups, bias period 8) %0d cycles",
                 errors ? "FAIL" : "PASS", ncmd, errors, epi_cycles);
        $finish;
    end

    initial begin #200_000_000; $display("TB FAIL: timeout after %0d commands", ncmd); $finish; end
endmodule
