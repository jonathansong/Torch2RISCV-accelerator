// Runs ddr_test firmware on picorv32_axi (same parameters as
// scripts/pico_processor.tcl) against an AXI4-Lite memory model with the
// RISC-V memory map of the overlay:
//   0xC0000000  8 KB   program BRAM (+ mailbox at 0xC0001F00)
//   DDR_BASE    16 KB  window standing in for PS DDR via S_AXI_HP0
// The initial block plays the ARM side of tests/ddr_access/ddr_test.py.
`timescale 1ns / 1ps

module tb_ddr_test;
    localparam [31:0] BRAM_BASE  = 32'hC000_0000;
    localparam        BRAM_WORDS = 2048;
    localparam [31:0] DDR_BASE   = 32'h1F00_0000;
    localparam        DDR_WORDS  = 4096;

    localparam        N_WORDS    = 256;
    localparam [31:0] SRC        = DDR_BASE;
    localparam [31:0] DST        = DDR_BASE + 32'h2000;
    localparam        SUB_BYTES  = 64;
    localparam        SUB_HALVES = 32;
    localparam        DST_WORDS  = N_WORDS + (SUB_BYTES + 2 * SUB_HALVES) / 4;
    localparam        GUARD      = 4;
    localparam [31:0] MBOX       = 32'h1F00 / 4;   // word index in BRAM

    reg clk = 0, resetn = 0;
    always #10 clk = ~clk;  // 50 MHz, like subprocessorClk

    wire        trap;
    wire        awvalid, wvalid, bready, arvalid, rready;
    reg         awready = 0, wready = 0, bvalid = 0, arready = 0, rvalid = 0;
    wire [31:0] awaddr, wdata, araddr;
    wire [ 3:0] wstrb;
    wire [ 2:0] awprot, arprot;
    reg  [31:0] rdata;

    picorv32_axi #(
        .COMPRESSED_ISA (1),
        .ENABLE_MUL     (1),
        .ENABLE_DIV     (1),
        .ENABLE_PCPI    (1),
        .PROGADDR_RESET (32'hC000_0000),
        .STACKADDR      (32'hC000_2000)
    ) dut (
        .clk(clk), .resetn(resetn), .trap(trap),
        .mem_axi_awvalid(awvalid), .mem_axi_awready(awready),
        .mem_axi_awaddr(awaddr),   .mem_axi_awprot(awprot),
        .mem_axi_wvalid(wvalid),   .mem_axi_wready(wready),
        .mem_axi_wdata(wdata),     .mem_axi_wstrb(wstrb),
        .mem_axi_bvalid(bvalid),   .mem_axi_bready(bready),
        .mem_axi_arvalid(arvalid), .mem_axi_arready(arready),
        .mem_axi_araddr(araddr),   .mem_axi_arprot(arprot),
        .mem_axi_rvalid(rvalid),   .mem_axi_rready(rready),
        .mem_axi_rdata(rdata),
        // PCPI unconnected in the overlay (tied to 0 by BD)
        .pcpi_wr(1'b0), .pcpi_rd(32'b0), .pcpi_wait(1'b0), .pcpi_ready(1'b0),
        .irq(32'b0)
    );

    reg [31:0] bram [0:BRAM_WORDS-1];
    reg [31:0] ddr  [0:DDR_WORDS-1];

    // ------------------------------------------------------------ memory
    function in_bram(input [31:0] a); in_bram = a >= BRAM_BASE && a < BRAM_BASE + 4*BRAM_WORDS; endfunction
    function in_ddr (input [31:0] a); in_ddr  = a >= DDR_BASE  && a < DDR_BASE  + 4*DDR_WORDS;  endfunction

    reg [31:0] seed = 1;
    function [1:0] lat; input dummy; begin seed = seed * 1103515245 + 12345; lat = seed[17:16]; end endfunction

    integer rd_wait = -1, wr_wait = -1;
    reg [31:0] rd_addr, wr_addr, wr_data;
    reg [3:0]  wr_strb;
    reg        aw_seen = 0, w_seen = 0;

    task store(input [31:0] a, input [31:0] d, input [3:0] s);
        integer k;
        begin
            if (in_bram(a)) begin
                for (k = 0; k < 4; k = k + 1)
                    if (s[k]) bram[(a - BRAM_BASE) >> 2][8*k +: 8] = d[8*k +: 8];
            end else if (in_ddr(a)) begin
                for (k = 0; k < 4; k = k + 1)
                    if (s[k]) ddr[(a - DDR_BASE) >> 2][8*k +: 8] = d[8*k +: 8];
            end else begin
                $display("FATAL: store to unmapped address %08x", a); $finish;
            end
        end
    endtask

    always @(posedge clk) begin
        arready <= 0; awready <= 0; wready <= 0;

        // read channel
        if (rvalid && rready) rvalid <= 0;
        if (arvalid && !arready && rd_wait < 0 && !rvalid) begin
            arready <= 1; rd_addr <= araddr; rd_wait <= lat(0);
        end
        if (rd_wait == 0) begin
            if (in_bram(rd_addr))     rdata <= bram[(rd_addr - BRAM_BASE) >> 2];
            else if (in_ddr(rd_addr)) rdata <= ddr [(rd_addr - DDR_BASE)  >> 2];
            else begin $display("FATAL: load from unmapped address %08x", rd_addr); $finish; end
            rvalid <= 1;
        end
        if (rd_wait >= 0) rd_wait <= rd_wait - 1;

        // write channels
        if (bvalid && bready) bvalid <= 0;
        if (awvalid && !awready && !aw_seen) begin awready <= 1; aw_seen <= 1; wr_addr <= awaddr; end
        if (wvalid && !wready && !w_seen) begin wready <= 1; w_seen <= 1; wr_data <= wdata; wr_strb <= wstrb; end
        if (aw_seen && w_seen && wr_wait < 0 && !bvalid) wr_wait <= lat(0);
        if (wr_wait == 0) begin
            store(wr_addr, wr_data, wr_strb);
            bvalid <= 1; aw_seen <= 0; w_seen <= 0;
        end
        if (wr_wait >= 0) wr_wait <= wr_wait - 1;
    end

    // ------------------------------------------------------ "ARM" side
    integer i, errors, cycles;
    reg [31:0] src_val [0:N_WORDS-1];
    reg [31:0] sum, got, expect;
    reg [7:0]  byte_got;
    reg [15:0] half_got;

    initial begin
        for (i = 0; i < BRAM_WORDS; i = i + 1) bram[i] = 0;
        $readmemh("ddr_test.hex", bram);
        for (i = 0; i < DDR_WORDS; i = i + 1) ddr[i] = 32'hDEADBEEF;

        sum = 0;
        for (i = 0; i < N_WORDS; i = i + 1) begin
            src_val[i] = 32'h9E3779B9 * (i + 1) ^ (i << 7);
            ddr[(SRC - DDR_BASE) / 4 + i] = src_val[i];
            sum = sum + src_val[i];
        end

        bram[MBOX + 0] = 0;
        bram[MBOX + 1] = SRC;
        bram[MBOX + 2] = DST;
        bram[MBOX + 3] = N_WORDS;

        repeat (20) @(posedge clk);
        resetn <= 1;

        cycles = 0;
        while (!trap && cycles < 500000) begin @(posedge clk); cycles = cycles + 1; end
        repeat (5) @(posedge clk);

        errors = 0;
        if (!trap) begin $display("TB ERROR: no trap after %0d cycles", cycles); errors = errors + 1; end
        if (bram[MBOX] !== 32'h600D600D) begin $display("TB ERROR: status %08x", bram[MBOX]); errors = errors + 1; end
        if (bram[MBOX + 4] !== sum) begin $display("TB ERROR: src_sum %08x, expected %08x", bram[MBOX + 4], sum); errors = errors + 1; end
        if (bram[MBOX + 5] !== 0) begin $display("TB ERROR: firmware self-check errors = %0d", bram[MBOX + 5]); errors = errors + 1; end

        for (i = 0; i < N_WORDS; i = i + 1) begin
            got = ddr[(DST - DDR_BASE) / 4 + i];
            expect = src_val[i] * 3 + i;
            if (got !== expect) begin
                if (errors < 10) $display("TB ERROR: dst[%0d] = %08x, expected %08x", i, got, expect);
                errors = errors + 1;
            end
        end
        for (i = 0; i < SUB_BYTES; i = i + 1) begin
            got = ddr[(DST - DDR_BASE) / 4 + N_WORDS + i / 4];
            byte_got = got[8 * (i % 4) +: 8];
            if (byte_got !== (i ^ 8'h5A)) begin
                if (errors < 10) $display("TB ERROR: byte[%0d] = %02x", i, byte_got);
                errors = errors + 1;
            end
        end
        for (i = 0; i < SUB_HALVES; i = i + 1) begin
            got = ddr[(DST - DDR_BASE) / 4 + N_WORDS + SUB_BYTES / 4 + i / 2];
            half_got = got[16 * (i % 2) +: 16];
            if (half_got !== 16'hA000 + 3 * i) begin
                if (errors < 10) $display("TB ERROR: half[%0d] = %04x", i, half_got);
                errors = errors + 1;
            end
        end
        for (i = 0; i < GUARD; i = i + 1)
            if (ddr[(DST - DDR_BASE) / 4 + DST_WORDS + i] !== 32'hDEADBEEF) begin
                $display("TB ERROR: guard word %0d overwritten", i); errors = errors + 1;
            end

        $display("TB %s: %0d errors, firmware cycles = %0d, total cycles to trap = %0d",
                 errors ? "FAIL" : "PASS", errors, bram[MBOX + 6], cycles);
        $finish;
    end
endmodule
