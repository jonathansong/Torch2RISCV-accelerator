// Unit test of the L2 fp32 vector engine / TRANSPOSE (sa_vefp) against the
// functional simulator: per case (sim/gen_vefp_cases.py) the memories are
// loaded, one command runs, and SPAD_A / SPAD_B / ACC words 0 .. NW-1 must
// equal the simulator's.
`timescale 1ns / 1ps
`include "sa_macros.vh"

module tb_sa_vefp;
    `include "sa_defs.vh"
    `include "vefp_cases.vh"
    parameter  integer D  = 8;
    parameter  integer FL = D / 2;              // physical fp lanes (D / 2 or D)
    localparam integer SW = 131072 / D, CW = 262144 / (4 * D);
    localparam integer SAW = $clog2(SW), CAW = $clog2(CW);

    reg clk = 0, resetn = 0;
    always #10 clk = ~clk;
    integer errors = 0, i, w, bad, cyc;

    reg         cmd_valid = 0;
    wire        cmd_ready, done, busy;
    reg  [31:0] c_s1, c_s2, c_dst, c_imm, c_a, c_b;
    reg  [15:0] c_g, c_p2, c_rl, c_vl, c_p1, c_s;
    reg  [7:0]  c_op;
    reg  [5:0]  c_ty;
    reg  [10:0] c_fl;
    wire        sa_en, sb_en, ac_en;
    wire [D-1:0] sa_we, sb_we;
    wire [4*D-1:0] ac_we;
    wire [SAW-1:0] sa_addr, sb_addr;
    wire [CAW-1:0] ac_addr;
    wire [8*D-1:0] sa_din, sb_din, sa_dout, sb_dout;
    wire [32*D-1:0] ac_din, ac_dout;

    sa_vefp #(.D(D), .FL(FL), .SPAD_AW(SAW), .ACC_AW(CAW)) dut (
        .clk(clk), .resetn(resetn), .cmd_valid(cmd_valid), .cmd_ready(cmd_ready),
        .cmd_src1(c_s1), .cmd_src2(c_s2), .cmd_dst(c_dst), .cmd_groups(c_g), .cmd_op(c_op), .cmd_types(c_ty),
        .cmd_mod(c_p2), .cmd_flags(c_fl), .cmd_imm(c_imm), .cmd_a(c_a), .cmd_b(c_b), .cmd_rowlen(c_rl),
        .cmd_vld(c_vl), .cmd_p1(c_p1), .cmd_s(c_s), .done(done), .busy(busy),
        .sa_en(sa_en), .sa_we(sa_we), .sa_addr(sa_addr), .sa_din(sa_din), .sa_dout(sa_dout),
        .sb_en(sb_en), .sb_we(sb_we), .sb_addr(sb_addr), .sb_din(sb_din), .sb_dout(sb_dout),
        .ac_en(ac_en), .ac_we(ac_we), .ac_addr(ac_addr), .ac_din(ac_din), .ac_dout(ac_dout), .perf_ev());

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

    reg [8*D-1:0]  ia [0:NW-1], ib [0:NW-1], ea [0:NW-1], eb [0:NW-1];
    reg [32*D-1:0] ic [0:NW-1], ec [0:NW-1];
    reg [8*D-1:0]  ga, gb;
    reg [32*D-1:0] gc;
    reg [8*64-1:0] fname;

    initial begin
        repeat (3) @(posedge clk);
        resetn = 1;
        for (i = 0; i < NVC; i = i + 1) begin
            $sformat(fname, "vc%0d_a.hex", i); $readmemh(fname, ia);
            $sformat(fname, "vc%0d_b.hex", i); $readmemh(fname, ib);
            $sformat(fname, "vc%0d_c.hex", i); $readmemh(fname, ic);
            $sformat(fname, "vc%0d_A.hex", i); $readmemh(fname, ea);
            $sformat(fname, "vc%0d_B.hex", i); $readmemh(fname, eb);
            $sformat(fname, "vc%0d_C.hex", i); $readmemh(fname, ec);
            for (w = 0; w < NW; w = w + 1) begin
                spad_a.bank[0].ram.ram_block[w] = ia[w];
                spad_b.bank[0].ram.ram_block[w] = ib[w];
                accm.bank[0].ram.ram_block[w]   = ic[w];
            end
            vc_cmd(i, c_s1, c_s2, c_dst, c_g, c_op, c_ty, c_p2, c_fl, c_imm, c_a, c_b, c_rl, c_vl, c_p1, c_s);
            @(posedge clk); #1 cmd_valid = 1;
            @(posedge clk); while (!cmd_ready) @(posedge clk);
            #1 cmd_valid = 0;
            cyc = 0;
            while (!done && cyc < 200000) begin @(posedge clk); cyc = cyc + 1; end
            if (!done) begin $display("TB ERROR case %0d: no done", i); errors = errors + 1; end
            repeat (2) @(posedge clk);
            bad = 0;
            for (w = 0; w < NW; w = w + 1) begin
                ga = spad_a.bank[0].ram.ram_block[w]; gb = spad_b.bank[0].ram.ram_block[w];
                gc = accm.bank[0].ram.ram_block[w];
                if (ga !== ea[w] || gb !== eb[w] || gc !== ec[w]) begin
                    bad = bad + 1;
                    if (bad <= 3) $display("TB ERROR case %0d word %0d: A %h/%h B %h/%h C %h / %h", i, w,
                                           ga, ea[w], gb, eb[w], gc, ec[w]);
                end
            end
            errors = errors + (bad != 0);
            $display("TB   -> %0d cycles, %0d words differ", cyc, bad);
        end
        $display("TB %s: %0d fp VE cases, %0d failing", errors ? "FAIL" : "PASS", NVC, errors);
        $finish;
    end
endmodule
