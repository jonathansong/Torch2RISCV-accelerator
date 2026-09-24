// Unit test for sa_ex: K-streaming matmul out of SPAD into ACC, checked
// against a behavioral golden model. Covers Kt from 1 to 64, operand strips
// crossing the bank boundary, back-to-back commands (drain overlapping the
// next stream), accumulate mode and int8 extremes.
`timescale 1ns / 1ps

module tb_sa_ex;
    parameter  integer D          = 8;             // make sim TB=tb_sa_ex GEN=D=16
    localparam integer SPAD_WORDS = 131072 / D;
    localparam integer ACC_WORDS  = 262144 / (4 * D);
    localparam integer SB         = SPAD_WORDS / 2;  // first word of bank 1
    localparam integer CB         = ACC_WORDS / 2;
    localparam integer SAW        = $clog2(SPAD_WORDS);
    localparam integer CAW        = $clog2(ACC_WORDS);
    localparam integer MAXK       = 512;
    localparam integer NCMD       = 12;

    reg clk = 0, resetn = 0;
    always #10 clk = ~clk;

    integer errors = 0;
    integer seed = 4242;

    // ------------------------------------------------------------ DUT
    reg          cmd_valid = 0;
    wire         cmd_ready, done, busy;
    reg  [15:0]  cmd_a, cmd_b, cmd_c;
    reg  [11:0]  cmd_kt;
    reg          cmd_acc;
    reg  [11:0]  cmd_rep = 0;
    reg  [15:0]  cmd_bstep = 0, cmd_cstep = 0, cmd_crow = 0;
    wire         sa_en, sb_en, acc_en;
    wire [SAW-1:0] sa_addr, sb_addr;
    wire [CAW-1:0] acc_addr;
    wire [8*D-1:0] sa_dout, sb_dout;
    wire [4*D-1:0] acc_we;
    wire [32*D-1:0] acc_din, acc_dout;

    sa_ex #(.D(D), .DSP_COLS(D), .SPAD_AW(SAW), .ACC_AW(CAW)) dut (
        .clk(clk), .resetn(resetn),
        .cmd_valid(cmd_valid), .cmd_ready(cmd_ready), .cmd_a(cmd_a), .cmd_b(cmd_b),
        .cmd_c(cmd_c), .cmd_kt(cmd_kt), .cmd_acc(cmd_acc),
        .cmd_rep(cmd_rep), .cmd_bstep(cmd_bstep), .cmd_cstep(cmd_cstep), .cmd_crow(cmd_crow),
        .done(done), .busy(busy),
        .sa_en(sa_en), .sa_addr(sa_addr), .sa_dout(sa_dout),
        .sb_en(sb_en), .sb_addr(sb_addr), .sb_dout(sb_dout),
        .acc_en(acc_en), .acc_we(acc_we), .acc_addr(acc_addr), .acc_din(acc_din), .acc_dout(acc_dout));

    // testbench owns side A of the SPADs and side B of ACC
    reg            ta_en = 0, tb_en = 0, tc_en = 0;
    reg  [D-1:0]   ta_we = 0, tb_we = 0;
    reg  [4*D-1:0] tc_we = 0;
    reg  [SAW-1:0] ta_addr = 0, tb_addr = 0;
    reg  [CAW-1:0] tc_addr = 0;
    reg  [8*D-1:0] ta_din = 0, tb_din = 0;
    reg  [32*D-1:0] tc_din = 0;
    wire [8*D-1:0] ta_dout, tb_dout;
    wire [32*D-1:0] tc_dout;

    sa_bankmem #(.W(8*D), .DEPTH(SPAD_WORDS)) spad_a (
        .clk(clk), .a_en(ta_en), .a_we(ta_we), .a_addr(ta_addr), .a_din(ta_din), .a_dout(ta_dout),
        .b_en(sa_en), .b_we({D{1'b0}}), .b_addr(sa_addr), .b_din({8*D{1'b0}}), .b_dout(sa_dout));
    sa_bankmem #(.W(8*D), .DEPTH(SPAD_WORDS)) spad_b (
        .clk(clk), .a_en(tb_en), .a_we(tb_we), .a_addr(tb_addr), .a_din(tb_din), .a_dout(tb_dout),
        .b_en(sb_en), .b_we({D{1'b0}}), .b_addr(sb_addr), .b_din({8*D{1'b0}}), .b_dout(sb_dout));
    sa_bankmem #(.W(32*D), .DEPTH(ACC_WORDS)) accm (
        .clk(clk), .a_en(acc_en), .a_we(acc_we), .a_addr(acc_addr), .a_din(acc_din), .a_dout(acc_dout),
        .b_en(tc_en), .b_we(tc_we), .b_addr(tc_addr), .b_din(tc_din), .b_dout(tc_dout));

    // ------------------------------------------------------------ helpers
    task wr_spad(input which, input integer word, input [8*D-1:0] data);
        begin
            @(posedge clk); #1;
            if (which == 0) begin ta_en = 1; ta_we = {D{1'b1}}; ta_addr = word; ta_din = data; end
            else            begin tb_en = 1; tb_we = {D{1'b1}}; tb_addr = word; tb_din = data; end
            @(posedge clk); #1 ta_en = 0; tb_en = 0; ta_we = 0; tb_we = 0;
        end
    endtask

    task wr_acc(input integer word, input [32*D-1:0] data);
        begin
            @(posedge clk); #1 tc_en = 1; tc_we = {4*D{1'b1}}; tc_addr = word; tc_din = data;
            @(posedge clk); #1 tc_en = 0; tc_we = 0;
        end
    endtask

    reg [32*D-1:0] rd_word;
    task rd_acc(input integer word);
        begin
            @(posedge clk); #1 tc_en = 1; tc_we = 0; tc_addr = word;
            @(posedge clk); #1 tc_en = 0;
            rd_word = tc_dout;
        end
    endtask

    // per-command operands and expected result
    reg signed [7:0]  A   [0:NCMD-1][0:D-1][0:MAXK-1];
    reg signed [7:0]  B   [0:NCMD-1][0:MAXK-1][0:D-1];
    reg signed [31:0] PRE [0:NCMD-1][0:D-1][0:D-1];
    integer ca [0:NCMD-1], cb [0:NCMD-1], cc [0:NCMD-1], ckt [0:NCMD-1], cacc [0:NCMD-1];

    function signed [7:0] rnd8(input integer mode);
        rnd8 = mode == 1 ? -8'sd128 : mode == 2 ? 8'sd127 : $random(seed);
    endfunction

    // fill operands of command n and write them into the SPADs in the layouts of §3.4
    integer i, j, k, w;
    reg [8*D-1:0] word;
    task make_cmd(input integer n, input integer a, input integer b, input integer cw,
                  input integer kt, input integer acc, input integer amode, input integer bmode);
        begin
            ca[n] = a; cb[n] = b; cc[n] = cw; ckt[n] = kt; cacc[n] = acc;
            for (i = 0; i < D; i = i + 1)
                for (k = 0; k < kt * D; k = k + 1) begin
                    A[n][i][k] = rnd8(amode);
                    B[n][k][i] = rnd8(bmode);
                end
            for (w = 0; w < kt; w = w + 1)                       // A: word w*D+g = A[g][wD..]
                for (i = 0; i < D; i = i + 1) begin
                    for (j = 0; j < D; j = j + 1) word[8*j +: 8] = A[n][i][w*D + j];
                    wr_spad(0, a + w*D + i, word);
                end
            for (k = 0; k < kt * D; k = k + 1) begin             // B: word k = B[k][0..D-1]
                for (j = 0; j < D; j = j + 1) word[8*j +: 8] = B[n][k][j];
                wr_spad(1, b + k, word);
            end
            for (i = 0; i < D; i = i + 1) begin
                for (j = 0; j < D; j = j + 1) begin
                    PRE[n][i][j] = acc ? $random(seed) : 0;
                    rd_word[32*j +: 32] = PRE[n][i][j];
                end
                wr_acc(cw + i, rd_word);
            end
        end
    endtask

    task issue(input integer n);
        begin
            @(posedge clk); #1;
            cmd_valid = 1; cmd_a = ca[n]; cmd_b = cb[n]; cmd_c = cc[n]; cmd_kt = ckt[n]; cmd_acc = cacc[n];
            @(posedge clk);
            while (!cmd_ready) @(posedge clk);
            #1 cmd_valid = 0;
        end
    endtask

    reg signed [31:0] expect, got;
    task check(input integer n);
        begin
            for (i = 0; i < D; i = i + 1) begin
                rd_acc(cc[n] + i);
                for (j = 0; j < D; j = j + 1) begin
                    expect = PRE[n][i][j];
                    for (k = 0; k < ckt[n] * D; k = k + 1) expect = expect + A[n][i][k] * B[n][k][j];
                    got = rd_word[32*j +: 32];
                    if (got !== expect) begin
                        errors = errors + 1;
                        if (errors <= 10)
                            $display("TB ERROR cmd %0d (Kt %0d acc %0d): C[%0d][%0d] = %0d, expected %0d",
                                     n, ckt[n], cacc[n], i, j, got, expect);
                    end
                end
            end
        end
    endtask

    // ---- M2: one command, R output tiles (shared A strip, B strip per tile)
    localparam integer MAXR = 8, MAXRK = 64;
    reg signed [7:0]  RA [0:D-1][0:MAXRK-1];
    reg signed [7:0]  RB [0:MAXR-1][0:MAXRK-1][0:D-1];
    reg signed [31:0] RPRE [0:MAXR-1][0:D-1][0:D-1];
    integer ra, rb, rbs, rc, rcs, rcr, rkt, rR, racc, rr;
    task make_rep(input integer a, input integer b, input integer bstep, input integer c,
                  input integer cstep, input integer crow, input integer kt, input integer R, input integer acc);
        begin
            ra = a; rb = b; rbs = bstep; rc = c; rcs = cstep; rcr = crow; rkt = kt; rR = R; racc = acc;
            for (i = 0; i < D; i = i + 1) for (k = 0; k < kt * D; k = k + 1) RA[i][k] = $random(seed);
            for (w = 0; w < kt; w = w + 1)
                for (i = 0; i < D; i = i + 1) begin
                    for (j = 0; j < D; j = j + 1) word[8*j +: 8] = RA[i][w*D + j];
                    wr_spad(0, a + w*D + i, word);
                end
            for (rr = 0; rr < R; rr = rr + 1) begin
                for (k = 0; k < kt * D; k = k + 1) begin
                    for (j = 0; j < D; j = j + 1) begin RB[rr][k][j] = $random(seed); word[8*j +: 8] = RB[rr][k][j]; end
                    wr_spad(1, b + rr*bstep + k, word);
                end
                for (i = 0; i < D; i = i + 1) begin
                    for (j = 0; j < D; j = j + 1) begin
                        RPRE[rr][i][j] = acc ? $random(seed) : 0;
                        rd_word[32*j +: 32] = RPRE[rr][i][j];
                    end
                    wr_acc(c + rr*cstep + i*crow, rd_word);
                end
            end
        end
    endtask
    task issue_rep;
        begin
            @(posedge clk); #1;
            cmd_valid = 1; cmd_a = ra; cmd_b = rb; cmd_c = rc; cmd_kt = rkt; cmd_acc = racc;
            cmd_rep = rR - 1; cmd_bstep = rbs; cmd_cstep = rcs; cmd_crow = rcr;
            @(posedge clk);
            while (!cmd_ready) @(posedge clk);
            #1 cmd_valid = 0; cmd_rep = 0; cmd_bstep = 0; cmd_cstep = 0; cmd_crow = 0;
        end
    endtask
    task check_rep;
        begin
            for (rr = 0; rr < rR; rr = rr + 1)
                for (i = 0; i < D; i = i + 1) begin
                    rd_acc(rc + rr*rcs + i*rcr);
                    for (j = 0; j < D; j = j + 1) begin
                        expect = RPRE[rr][i][j];
                        for (k = 0; k < rkt * D; k = k + 1) expect = expect + RA[i][k] * RB[rr][k][j];
                        got = rd_word[32*j +: 32];
                        if (got !== expect) begin
                            errors = errors + 1;
                            if (errors <= 10) $display("TB ERROR repeat R=%0d tile %0d: C[%0d][%0d] = %0d, expected %0d",
                                                       rR, rr, i, j, got, expect);
                        end
                    end
                end
        end
    endtask

    integer dones = 0;
    always @(posedge clk) if (done) dones = dones + 1;

    // the D = 8 addresses scaled to the memory depth, and Kt scaled so a
    // strip (Kt*D words) shrinks with it and strips never overlap
    function integer sw(input integer w); sw = w * SPAD_WORDS / 16384; endfunction
    function integer ktd(input integer kt); ktd = kt * 64 / (D * D) < 1 ? 1 : kt * 64 / (D * D); endfunction

    integer n, t0, t1, tr0, single_cycles, rep_cycles;
    initial begin
        repeat (5) @(posedge clk);
        #1 resetn = 1;

        // --- single 8x8x8 job, timing
        make_cmd(0, 0, 0, 0, 1, 0, 0, 0);
        t0 = $time;
        issue(0);
        while (dones < 1) @(posedge clk);
        single_cycles = ($time - t0) / 20;
        check(0);

        // --- batch of commands issued back-to-back; strips crossing the bank
        //     boundary (SPAD bank = 8192 words, ACC bank = 4096 words)
        //     boundary (SPAD bank = SB words, ACC bank = CB words; D = 8: 8192 / 4096)
        make_cmd(1, sw(100),  sw(200),  2*D, ktd(2),  0, 0, 0);
        make_cmd(2, SB - 3*D, SB - 5, CB - 3, ktd(5), 0, 0, 0);     // A, B and C straddle banks
        make_cmd(3, sw(9000), sw(300),  CB + 104, ktd(16), 0, 0, 0);
        make_cmd(4, sw(2000), sw(9100), 4*D, ktd(64), 0, 0, 0);     // K = 512
        make_cmd(5, sw(5000), sw(5000), 6*D, ktd(3),  1, 0, 0);     // accumulate
        make_cmd(6, sw(6000), sw(6000), 8*D, ktd(64), 0, 1, 1);     // -128 * -128, K = 512
        make_cmd(7, sw(7000), sw(7000), 10*D, ktd(7),  0, 1, 2);    // -128 * 127
        make_cmd(8, sw(12000), sw(12000), 12*D, 1, 1, 2, 2);        // accumulate, Kt = 1
        make_cmd(9, sw(13000), sw(13000), 14*D, 1, 0, 0, 0);
        make_cmd(10, sw(14000), sw(14000), 16*D, ktd(2), 0, 0, 0);
        make_cmd(11, sw(15000), sw(15000), 18*D, 1, 1, 0, 0);
        t0 = $time;
        for (n = 1; n < NCMD; n = n + 1) issue(n);
        while (dones < NCMD) @(posedge clk);
        t1 = $time;
        for (n = 1; n < NCMD; n = n + 1) check(n);

        // --- M2: repeat commands (one done per command)
        make_rep(sw(1000), sw(2000), 64, CB - 1096, 1, 8, 64 / D, 8, 0); // resident-B strip, row-major C strip
        tr0 = $time; n = dones;
        issue_rep;
        while (dones < n + 1) @(posedge clk);
        rep_cycles = ($time - tr0) / 20;
        check_rep;
        if (dones != n + 1) begin errors = errors + 1; $display("TB ERROR: done count for repeat"); end
        make_rep(sw(4000), sw(5000), 100, CB - 596, D, 1, 2, 3, 1); // gaps between B strips, tile blocks, accumulate
        issue_rep;
        while (dones < n + 2) @(posedge clk);
        check_rep;
        make_rep(SB - 16, SB - 40, 2*D, CB - 20, 1, 5, 2, 5, 0);   // across bank boundaries
        issue_rep;
        while (dones < n + 3) @(posedge clk);
        check_rep;

        $display("TB %s: %0d commands + 3 repeat commands, %0d errors; single 8x8x8 %0d cycles, batch %0d cycles, 8 tiles x Kt 8 in one command %0d cycles",
                 errors ? "FAIL" : "PASS", NCMD, errors, single_cycles, (t1 - t0) / 20, rep_cycles);
        $finish;
    end

    initial begin
        #20_000_000;
        $display("TB FAIL: timeout (%0d of %0d commands done)", dones, NCMD);
        $finish;
    end
endmodule
