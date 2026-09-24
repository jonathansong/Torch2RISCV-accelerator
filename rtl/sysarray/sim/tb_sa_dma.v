// Unit test for sa_ld / sa_st against an AXI memory model with NP ports,
// several bursts outstanding per port, random stalls, SLVERR injection and
// protocol checks. Expected local/DDR contents come from an independent
// byte-by-byte reference of the transfer shapes; every command's target
// region plus neighbours is compared, so stray writes are caught too.
`timescale 1ns / 1ps
`include "sa_macros.vh"

module tb_sa_dma;
    parameter integer NP = 1;

    `include "sa_defs.vh"
    localparam integer D          = 8;
    localparam integer SPAD_WORDS = 16384;
    localparam integer ACC_WORDS  = 8192;
    localparam integer SAW        = $clog2(SPAD_WORDS);
    localparam integer CAW        = $clog2(ACC_WORDS);
    localparam [31:0]  DDR_BASE   = 32'h2000_0000;
    localparam integer DDR_BYTES  = 65536;

    reg clk = 0, resetn = 0;
    always #10 clk = ~clk;

    integer errors = 0;
    integer seed = 99;
    function integer rnd(input integer max);
        rnd = $unsigned($random(seed)) % (max + 1);
    endfunction
    task fail(input [8*80-1:0] msg);
        begin errors = errors + 1; if (errors <= 15) $display("TB ERROR @%0t: %0s", $time, msg); end
    endtask

    // --------------------------------------------------------------- DUTs
    reg         ld_valid = 0, st_valid = 0;
    wire        ld_ready, st_ready, ld_done, st_done, ld_err, st_err, ld_busy, st_busy;
    reg  [31:0] c_ddr, c_pitch;
    reg  [3:0]  c_mem;
    reg  [15:0] c_word, c_rows, c_rb;
    reg  [1:0]  c_mode;

    wire        lw_en;
    wire [3:0]  lw_mem;
    wire [15:0] lw_word;
    wire [7:0]  lw_lane;
    wire [63:0] lw_data;
    wire        lr_en;
    wire [3:0]  lr_mem;
    wire [15:0] lr_word;
    wire [8*D-1:0]  lr_a, lr_b;
    wire [32*D-1:0] lr_c;

    wire [NP*32-1:0] araddr, awaddr;
    wire [NP*8-1:0]  arlen, awlen;
    wire [NP-1:0]    arvalid, rready, awvalid, wvalid, wlast, bready;
    reg  [NP-1:0]    arready = 0, rvalid = 0, rlast = 0, awready = 0, wready = 0, bvalid = 0;
    reg  [NP*64-1:0] rdata = 0;
    reg  [NP*2-1:0]  rresp = 0, bresp = 0;
    wire [NP*64-1:0] wdata;

    sa_ld #(.D(D), .NPORTS(NP)) ld (
        .clk(clk), .resetn(resetn), .cmd_valid(ld_valid), .cmd_ready(ld_ready),
        .cmd_ddr(c_ddr), .cmd_mem(c_mem), .cmd_word(c_word), .cmd_rows(c_rows),
        .cmd_row_bytes(c_rb), .cmd_pitch(c_pitch), .cmd_mode(c_mode),
        .done(ld_done), .err(ld_err), .busy(ld_busy),
        .lw_en(lw_en), .lw_mem(lw_mem), .lw_word(lw_word), .lw_lane(lw_lane), .lw_data(lw_data),
        .m_araddr(araddr), .m_arlen(arlen), .m_arvalid(arvalid), .m_arready(arready),
        .m_rdata(rdata), .m_rresp(rresp), .m_rlast(rlast), .m_rvalid(rvalid), .m_rready(rready));
    sa_st #(.D(D), .NPORTS(NP)) st (
        .clk(clk), .resetn(resetn), .cmd_valid(st_valid), .cmd_ready(st_ready),
        .cmd_ddr(c_ddr), .cmd_mem(c_mem), .cmd_word(c_word), .cmd_rows(c_rows),
        .cmd_row_bytes(c_rb), .cmd_pitch(c_pitch),
        .done(st_done), .err(st_err), .busy(st_busy),
        .lr_en(lr_en), .lr_mem(lr_mem), .lr_word(lr_word),
        .lr_spad_a(lr_a), .lr_spad_b(lr_b), .lr_acc(lr_c),
        .m_awaddr(awaddr), .m_awlen(awlen), .m_awvalid(awvalid), .m_awready(awready),
        .m_wdata(wdata), .m_wlast(wlast), .m_wvalid(wvalid), .m_wready(wready),
        .m_bresp(bresp), .m_bvalid(bvalid), .m_bready(bready));

    // ------------------------------------------------ local memories
    // SPAD side A: [0] LD write, [1] ST read; side B: testbench.
    // ACC  side A: testbench;           side B: [0] LD write, [1] ST read.
    reg            t_en = 0;
    reg  [3:0]     t_mem = 0;
    reg            t_we = 0;
    reg  [15:0]    t_word = 0;
    reg  [32*D-1:0] t_din = 0;
    wire [8*D-1:0]  ta_dout, tb_dout;
    wire [32*D-1:0] tc_dout;

    wire [8*D-1:0]  ta_unused_a, ta_unused_b;
    wire [32*D-1:0] tc_unused;

    function [8*D-1:0] spad_din(input [63:0] d); spad_din = {D/8{d}}; endfunction
    function [D-1:0]   spad_we(input [7:0] lane); spad_we = {8'hFF} << (8 * lane); endfunction
    function [4*D-1:0] acc_we(input [7:0] lane);  acc_we  = {{4*D-8{1'b0}}, 8'hFF} << (8 * lane); endfunction

    wire ld_a = lw_en && lw_mem == MEM_SPAD_A, ld_b = lw_en && lw_mem == MEM_SPAD_B, ld_c = lw_en && lw_mem == MEM_ACC;
    wire st_a = lr_en && lr_mem == MEM_SPAD_A, st_b = lr_en && lr_mem == MEM_SPAD_B, st_c = lr_en && lr_mem == MEM_ACC;

    sa_bankmem #(.W(8*D), .DEPTH(SPAD_WORDS), .NA(2), .NB(1)) spad_a (
        .clk(clk),
        .a_en({st_a, ld_a}), .a_we({{D{1'b0}}, spad_we(lw_lane)}),
        .a_addr({lr_word[SAW-1:0], lw_word[SAW-1:0]}), .a_din({{8*D{1'b0}}, spad_din(lw_data)}),
        .a_dout({lr_a, ta_unused_a}),
        .b_en(t_en && t_mem == MEM_SPAD_A), .b_we({D{t_we}}), .b_addr(t_word[SAW-1:0]),
        .b_din(t_din[8*D-1:0]), .b_dout(ta_dout));
    sa_bankmem #(.W(8*D), .DEPTH(SPAD_WORDS), .NA(2), .NB(1)) spad_b (
        .clk(clk),
        .a_en({st_b, ld_b}), .a_we({{D{1'b0}}, spad_we(lw_lane)}),
        .a_addr({lr_word[SAW-1:0], lw_word[SAW-1:0]}), .a_din({{8*D{1'b0}}, spad_din(lw_data)}),
        .a_dout({lr_b, ta_unused_b}),
        .b_en(t_en && t_mem == MEM_SPAD_B), .b_we({D{t_we}}), .b_addr(t_word[SAW-1:0]),
        .b_din(t_din[8*D-1:0]), .b_dout(tb_dout));
    sa_bankmem #(.W(32*D), .DEPTH(ACC_WORDS), .NA(1), .NB(2)) accm (
        .clk(clk),
        .a_en(t_en && t_mem == MEM_ACC), .a_we({4*D{t_we}}), .a_addr(t_word[CAW-1:0]),
        .a_din(t_din), .a_dout(tc_dout),
        .b_en({st_c, ld_c}), .b_we({{4*D{1'b0}}, acc_we(lw_lane)}),
        .b_addr({lr_word[CAW-1:0], lw_word[CAW-1:0]}), .b_din({{32*D{1'b0}}, {D/2{lw_data}}}),
        .b_dout({lr_c, tc_unused}));

    // shadows of the local memories (expected contents)
    reg [8*D-1:0]  sh_a [0:SPAD_WORDS-1];
    reg [8*D-1:0]  sh_b [0:SPAD_WORDS-1];
    reg [32*D-1:0] sh_c [0:ACC_WORDS-1];

    function integer wbytes(input [3:0] mem); wbytes = mem == MEM_ACC ? 4 * D : D; endfunction

    task t_access(input [3:0] mem, input integer word, input we, input [32*D-1:0] din);
        begin
            @(posedge clk); #1 t_en = 1; t_mem = mem; t_word = word; t_we = we; t_din = din;
            @(posedge clk); #1 t_en = 0; t_we = 0;
        end
    endtask

    reg [32*D-1:0] rnd_word;
    integer z;
    task init_region(input [3:0] mem, input integer from, input integer to);
        integer wd;
        begin
            for (wd = from; wd <= to; wd = wd + 1) begin
                for (z = 0; z < D; z = z + 1) rnd_word[32*z +: 32] = $random(seed);
                if (mem == MEM_SPAD_A) sh_a[wd] = rnd_word[8*D-1:0];
                if (mem == MEM_SPAD_B) sh_b[wd] = rnd_word[8*D-1:0];
                if (mem == MEM_ACC)    sh_c[wd] = rnd_word;
                t_access(mem, wd, 1, rnd_word);
            end
        end
    endtask

    task check_region(input [3:0] mem, input integer from, input integer to);
        integer wd;
        reg [32*D-1:0] got, exp;
        begin
            for (wd = from; wd <= to; wd = wd + 1) begin
                t_access(mem, wd, 0, 0);
                got = mem == MEM_SPAD_A ? ta_dout : mem == MEM_SPAD_B ? tb_dout : tc_dout;
                exp = mem == MEM_SPAD_A ? sh_a[wd] : mem == MEM_SPAD_B ? sh_b[wd] : sh_c[wd];
                if (got !== exp) begin
                    errors = errors + 1;
                    if (errors <= 15) $display("TB ERROR mem %0d word %0d: %h, expected %h", mem, wd, got, exp);
                end
            end
        end
    endtask

    // -------------------------------------------------------- DDR model
    reg [7:0] ddr    [0:DDR_BYTES-1];
    reg [7:0] ddr_sh [0:DDR_BYTES-1];
    reg       inj_rresp = 0, inj_bresp = 0;
    integer   ar_bursts = 0, aw_bursts = 0;

    task chk_burst(input [31:0] a, input [7:0] len);
        begin
            if (len > 15) fail("burst longer than 16 beats");
            if (a[2:0] != 0) fail("unaligned burst");
            if ((a & 32'hFFF) + (len + 1) * 8 > 32'h1000) fail("burst crosses 4 KB");
            if (a < DDR_BASE || a + (len + 1) * 8 > DDR_BASE + DDR_BYTES) fail("burst outside DDR model");
        end
    endtask

    genvar gp;
    generate
        for (gp = 0; gp < NP; gp = gp + 1) begin : port
            // read: AR queue, R served in order with random gaps
            reg [31:0] rq_a [0:15];
            reg [7:0]  rq_l [0:15];
            integer    rq_w = 0, rq_r = 0, rb, k;
            initial forever begin
                @(posedge clk);
                if (arvalid[gp] && arready[gp]) begin
                    chk_burst(araddr[32*gp +: 32], arlen[8*gp +: 8]);
                    rq_a[rq_w % 16] = araddr[32*gp +: 32]; rq_l[rq_w % 16] = arlen[8*gp +: 8];
                    rq_w = rq_w + 1; ar_bursts = ar_bursts + 1;
                end
                #1 arready[gp] = (rq_w - rq_r < 10) && rnd(3) != 0;
            end
            initial forever begin
                @(posedge clk);
                if (rq_r != rq_w) begin
                    for (rb = 0; rb <= rq_l[rq_r % 16]; rb = rb + 1) begin
                        repeat (rnd(3)) @(posedge clk);
                        #1;
                        for (k = 0; k < 8; k = k + 1)
                            rdata[64*gp + 8*k +: 8] = ddr[rq_a[rq_r % 16] - DDR_BASE + 8*rb + k];
                        rresp[2*gp +: 2] = inj_rresp && rb == 1 ? 2'b10 : 2'b00;
                        rlast[gp] = rb == rq_l[rq_r % 16];
                        rvalid[gp] = 1;
                        @(posedge clk);
                        while (!rready[gp]) @(posedge clk);
                        #1 rvalid[gp] = 0; rlast[gp] = 0;
                    end
                    rq_r = rq_r + 1;
                end
            end
            // write: AW queue, W consumed in AW order, B after each burst
            reg [31:0] wq_a [0:15];
            reg [7:0]  wq_l [0:15];
            integer    wq_w = 0, wq_r = 0, wb, j;
            initial forever begin
                @(posedge clk);
                if (awvalid[gp] && awready[gp]) begin
                    chk_burst(awaddr[32*gp +: 32], awlen[8*gp +: 8]);
                    wq_a[wq_w % 16] = awaddr[32*gp +: 32]; wq_l[wq_w % 16] = awlen[8*gp +: 8];
                    wq_w = wq_w + 1; aw_bursts = aw_bursts + 1;
                end
                if (wvalid[gp] && wq_w == wq_r) fail("W beat before its AW");
                #1 awready[gp] = (wq_w - wq_r < 10) && rnd(3) != 0;
            end
            initial forever begin
                @(posedge clk);
                if (wq_r != wq_w) begin
                    for (wb = 0; wb <= wq_l[wq_r % 16]; wb = wb + 1) begin
                        repeat (rnd(3)) @(posedge clk);
                        #1 wready[gp] = 1;
                        @(posedge clk);
                        while (!wvalid[gp]) @(posedge clk);
                        if (wlast[gp] != (wb == wq_l[wq_r % 16])) fail("WLAST on wrong beat");
                        for (j = 0; j < 8; j = j + 1)
                            ddr[wq_a[wq_r % 16] - DDR_BASE + 8*wb + j] = wdata[64*gp + 8*j +: 8];
                        #1 wready[gp] = 0;
                    end
                    repeat (rnd(4)) @(posedge clk);
                    #1 bresp[2*gp +: 2] = inj_bresp ? 2'b10 : 2'b00; bvalid[gp] = 1;
                    @(posedge clk);
                    while (!bready[gp]) @(posedge clk);
                    #1 bvalid[gp] = 0;
                    wq_r = wq_r + 1;
                end
            end
        end
    endgenerate

    // ------------------------------------------------------- commands
    task run_ld(input [31:0] ddr_a, input [3:0] mem, input integer word, input integer rows,
                input integer rb, input integer pitch, input integer mode);
        begin
            @(posedge clk); #1;
            c_ddr = ddr_a; c_mem = mem; c_word = word; c_rows = rows; c_rb = rb; c_pitch = pitch; c_mode = mode;
            ld_valid = 1;
            @(posedge clk); while (!ld_ready) @(posedge clk);
            #1 ld_valid = 0;
            @(posedge clk); while (!ld_done) @(posedge clk);
        end
    endtask

    task run_st(input [31:0] ddr_a, input [3:0] mem, input integer word, input integer rows,
                input integer rb, input integer pitch);
        begin
            @(posedge clk); #1;
            c_ddr = ddr_a; c_mem = mem; c_word = word; c_rows = rows; c_rb = rb; c_pitch = pitch; c_mode = 0;
            st_valid = 1;
            @(posedge clk); while (!st_ready) @(posedge clk);
            #1 st_valid = 0;
            @(posedge clk); while (!st_done) @(posedge clk);
        end
    endtask

    // reference: expected local contents after a load
    task ref_ld(input [31:0] ddr_a, input [3:0] mem, input integer word, input integer rows,
                input integer rb, input integer pitch, input integer mode);
        integer rr, bb, wd, byt, wb_, wpr;
        begin
            wb_ = wbytes(mem);
            wpr = (rb + wb_ - 1) / wb_;
            for (rr = 0; rr < rows; rr = rr + 1)
                for (bb = 0; bb < rb; bb = bb + 1) begin
                    wd  = mode ? word + (bb / wb_) * rows + rr : word + rr * wpr + bb / wb_;
                    byt = bb % wb_;
                    if (mem == MEM_SPAD_A) sh_a[wd][8*byt +: 8] = ddr[ddr_a - DDR_BASE + rr * pitch + bb];
                    if (mem == MEM_SPAD_B) sh_b[wd][8*byt +: 8] = ddr[ddr_a - DDR_BASE + rr * pitch + bb];
                    if (mem == MEM_ACC)    sh_c[wd][8*byt +: 8] = ddr[ddr_a - DDR_BASE + rr * pitch + bb];
                end
        end
    endtask

    // reference: expected DDR contents after a store
    task ref_st(input [31:0] ddr_a, input [3:0] mem, input integer word, input integer rows,
                input integer rb, input integer pitch);
        integer rr, bb, wd, byt, wb_, wpr;
        begin
            wb_ = wbytes(mem);
            wpr = (rb + wb_ - 1) / wb_;
            for (rr = 0; rr < rows; rr = rr + 1)
                for (bb = 0; bb < rb; bb = bb + 1) begin
                    wd  = word + rr * wpr + bb / wb_;
                    byt = bb % wb_;
                    ddr_sh[ddr_a - DDR_BASE + rr * pitch + bb] =
                        mem == MEM_SPAD_A ? sh_a[wd][8*byt +: 8] :
                        mem == MEM_SPAD_B ? sh_b[wd][8*byt +: 8] : sh_c[wd][8*byt +: 8];
                end
        end
    endtask

    task check_ddr;
        integer q, bad;
        begin
            bad = 0;
            for (q = 0; q < DDR_BYTES; q = q + 1)
                if (ddr[q] !== ddr_sh[q]) begin
                    bad = bad + 1;
                    if (bad <= 5) $display("TB ERROR ddr[+0x%0h] = %h, expected %h", q, ddr[q], ddr_sh[q]);
                end
            errors = errors + bad;
        end
    endtask

    // one load: prepare target region, run, update reference, compare with margin
    integer lo, hi, span;
    task do_ld(input [31:0] ddr_a, input [3:0] mem, input integer word, input integer rows,
               input integer rb, input integer pitch, input integer mode);
        begin
            span = mode ? ((rb + wbytes(mem) - 1) / wbytes(mem)) * rows
                        : rows * ((rb + wbytes(mem) - 1) / wbytes(mem));
            lo = word - 2; hi = word + span + 1;
            init_region(mem, lo, hi);
            run_ld(ddr_a, mem, word, rows, rb, pitch, mode);
            if (ld_err) fail("unexpected LD error");
            ref_ld(ddr_a, mem, word, rows, rb, pitch, mode);
            check_region(mem, lo, hi);
        end
    endtask

    task do_st(input [31:0] ddr_a, input [3:0] mem, input integer word, input integer rows,
               input integer rb, input integer pitch);
        begin
            span = rows * ((rb + wbytes(mem) - 1) / wbytes(mem));
            init_region(mem, word, word + span - 1);
            run_st(ddr_a, mem, word, rows, rb, pitch);
            if (st_err) fail("unexpected ST error");
            ref_st(ddr_a, mem, word, rows, rb, pitch);
            check_ddr;
        end
    endtask

    integer q;
    initial begin
        for (q = 0; q < DDR_BYTES; q = q + 1) begin ddr[q] = $random(seed); ddr_sh[q] = ddr[q]; end
        repeat (5) @(posedge clk);
        #1 resetn = 1;

        // ---- loads
        do_ld(DDR_BASE + 32'h0FE8, MEM_SPAD_A, 100, 5, 40, 48, 0);         // crosses 4 KB, 5 rows
        do_ld(DDR_BASE + 32'h1000, MEM_SPAD_B, 300, 8, 64, 104, 1);        // INTERLEAVE: A strip, K = 64
        do_ld(DDR_BASE + 32'h2000, MEM_SPAD_A, 8192 - 20, 8, 256, 256, 1); // INTERLEAVE across the bank boundary
        do_ld(DDR_BASE + 32'h3000, MEM_ACC, 40, 3, 32, 32, 0);            // one ACC word per row
        do_ld(DDR_BASE + 32'h3100, MEM_ACC, 60, 4, 24, 40, 0);            // partial ACC words
        do_ld(DDR_BASE + 32'h3200, MEM_ACC, 4096 - 3, 2, 72, 80, 0);      // 2.25 words per row, across banks
        do_ld(DDR_BASE + 32'h4008, MEM_SPAD_B, 8000, 1, 4160, 4160, 0);    // one long row, many bursts
        do_ld(DDR_BASE + 32'h6000, MEM_SPAD_A, 2000, 64, 8, 8, 0);         // B strip: 64 rows of one word

        // ---- stores
        do_st(DDR_BASE + 32'h8000, MEM_ACC, 200, 8, 32, 32);              // C tile
        do_st(DDR_BASE + 32'h8FE0, MEM_ACC, 300, 2, 64, 96);              // crosses 4 KB, 2 words per row
        do_st(DDR_BASE + 32'hA000, MEM_SPAD_A, 500, 4, 24, 40);           // gaps between rows untouched
        do_st(DDR_BASE + 32'hB008, MEM_SPAD_B, 8190, 1, 2056, 2056);   // long row across the bank boundary
        do_st(DDR_BASE + 32'hC000, MEM_ACC, 4000, 3, 40, 48);             // partial words

        // ---- errors, then recovery
        inj_rresp = 1;
        run_ld(DDR_BASE + 32'hD000, MEM_SPAD_A, 700, 2, 64, 64, 0);
        inj_rresp = 0;
        if (!ld_err) fail("LD SLVERR not reported");
        inj_bresp = 1;
        run_st(DDR_BASE + 32'hD800, MEM_SPAD_A, 700, 2, 64, 64);
        inj_bresp = 0;
        if (!st_err) fail("ST SLVERR not reported");
        // make the reference agree with whatever those two wrote, then recover
        for (q = 0; q < DDR_BYTES; q = q + 1) ddr_sh[q] = ddr[q];
        do_ld(DDR_BASE + 32'hE000, MEM_SPAD_A, 900, 4, 32, 32, 0);
        do_st(DDR_BASE + 32'hE800, MEM_SPAD_A, 900, 4, 32, 32);

        repeat (20) @(posedge clk);
        $display("TB %s: NP=%0d, %0d errors (%0d read / %0d write bursts)",
                 errors ? "FAIL" : "PASS", NP, errors, ar_bursts, aw_bursts);
        $finish;
    end

    initial begin
        #50_000_000;
        $display("TB FAIL: timeout");
        $finish;
    end
endmodule
