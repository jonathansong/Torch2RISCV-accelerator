// System test: firmware (fw.hex) running on picorv32_axi drives the real
// sa_unit through its CSRs or its PCPI port, as wired in the overlay; the
// unit's DMA port 0 reads/writes a DDR model.
//   default   : 32 independent 8x8x8 jobs (firmware/matmul CSR path or
//               firmware/matmul_insn mat_trigger path)
//   GEMM_TEST : firmware/gemm, tiled GEMMs on the funct7 = 1 ISA, int32 and
//               (M3) int8 output through the vector-engine epilogue
//   VEC_TEST  : firmware/vector, standalone vector operations (funct7 = 2)
//   BW_TEST   : firmware/bwtest, DMA bandwidth (simulated DDR model)
//   DESC_TEST : firmware/desc_run, GEMMs as descriptor lists built here (the
//               "ARM" side), one mat_submit each
//   RING_TEST : firmware/rt (L1), the resident runtime: submission ring,
//               completion records, notify_irq edges, errors, wrap-around
// Mirrors the overlay's RISC-V memory map:
//   0xC0000000  8 KB program BRAM (+ mailbox at 0xC0001F00)
//   0x80000000  matmul_unit CSRs
//   DDR         S_AXI_HP0 (identity-mapped physical addresses; same model)
// and the matmul DMA view of DDR (identity-mapped physical addresses).
// The "ARM" side (initial block) plays driver/pynq_matmul.py.
`timescale 1ns / 1ps

`include "sa_macros.vh"
`ifndef SIM_D
`define SIM_D 8          // array size: make sim SIM_D=16
`endif
module tb_system;
    `include "n_cases.vh"

    localparam [31:0] BRAM_BASE  = 32'hC000_0000;
    localparam        BRAM_WORDS = 2048;
    localparam [31:0] CSR_BASE   = 32'h8000_0000;
    localparam [31:0] DDR_BASE   = 32'h1800_0000;
    localparam        DDR_BYTES  = 327680;
    localparam [31:0] A_BASE     = DDR_BASE;             // NC * 64 B
    localparam [31:0] B_BASE     = DDR_BASE + 32'h1000;  // NC * 64 B
    localparam [31:0] C_BASE     = DDR_BASE + 32'h2000;  // NC * 256 B
    localparam [31:0] DESC_BASE  = DDR_BASE + 32'h5000;  // NC * 16 B (gap after C for the guard check)
    localparam [31:0] MBOX       = 32'h1F00 / 4;

    reg clk = 0, resetn = 0;
    always #10 clk = ~clk;   // 50 MHz, riscv_clk

    integer errors = 0;
    integer seed = 777;
    function integer rnd(input integer max);
        rnd = $unsigned($random(seed)) % (max + 1);
    endfunction

    // --------------------------------------------------------- PicoRV32
    wire        trap;
    wire        c_awvalid, c_wvalid, c_bready, c_arvalid, c_rready;
    wire [31:0] c_awaddr, c_wdata, c_araddr;
    wire [3:0]  c_wstrb;
    wire [2:0]  c_awprot, c_arprot;
    wire        c_awready, c_wready, c_bvalid, c_arready, c_rvalid;
    wire [31:0] c_rdata;

    wire        pcpi_valid, pcpi_wr, pcpi_wait, pcpi_ready;   // CPU <-> matmul_unit
    wire [31:0] pcpi_insn, pcpi_rs1, pcpi_rs2, pcpi_rd;

    picorv32_axi #(
        .COMPRESSED_ISA (1), .ENABLE_MUL (1), .ENABLE_DIV (1), .ENABLE_PCPI (1),
        .PROGADDR_RESET (32'hC000_0000), .STACKADDR (32'hC000_2000)
    ) cpu (
        .clk(clk), .resetn(resetn), .trap(trap),
        .mem_axi_awvalid(c_awvalid), .mem_axi_awready(c_awready),
        .mem_axi_awaddr(c_awaddr),   .mem_axi_awprot(c_awprot),
        .mem_axi_wvalid(c_wvalid),   .mem_axi_wready(c_wready),
        .mem_axi_wdata(c_wdata),     .mem_axi_wstrb(c_wstrb),
        .mem_axi_bvalid(c_bvalid),   .mem_axi_bready(c_bready),
        .mem_axi_arvalid(c_arvalid), .mem_axi_arready(c_arready),
        .mem_axi_araddr(c_araddr),   .mem_axi_arprot(c_arprot),
        .mem_axi_rvalid(c_rvalid),   .mem_axi_rready(c_rready),
        .mem_axi_rdata(c_rdata),
        .pcpi_valid(pcpi_valid), .pcpi_insn(pcpi_insn), .pcpi_rs1(pcpi_rs1), .pcpi_rs2(pcpi_rs2),
        .pcpi_wr(pcpi_wr), .pcpi_rd(pcpi_rd), .pcpi_wait(pcpi_wait), .pcpi_ready(pcpi_ready),
        .irq(32'b0)
    );

    // ------------------------------------ address decode: BRAM vs CSR
    // PicoRV32 keeps awaddr/araddr stable for the whole transaction.
    wire aw_csr = c_awaddr[31:12] == CSR_BASE[31:12];
    wire ar_csr = c_araddr[31:12] == CSR_BASE[31:12];

    // CPU-side CSR traffic (0 expected for the custom-instruction firmware)
    integer csr_accesses = 0;
    always @(posedge clk)
        if (resetn)
            csr_accesses = csr_accesses + (c_awvalid && c_awready && aw_csr) + (c_arvalid && c_arready && ar_csr);

    wire        mm_awready, mm_wready, mm_bvalid, mm_arready, mm_rvalid;
    wire [31:0] mm_rdata;
    wire [1:0]  mm_bresp, mm_rresp;
    reg         br_awready = 0, br_wready = 0, br_bvalid = 0, br_arready = 0, br_rvalid = 0;
    reg  [31:0] br_rdata;

    assign c_awready = aw_csr ? mm_awready : br_awready;
    assign c_wready  = aw_csr ? mm_wready  : br_wready;
    assign c_bvalid  = mm_bvalid | br_bvalid;
    assign c_arready = ar_csr ? mm_arready : br_arready;
    assign c_rvalid  = mm_rvalid | br_rvalid;
    assign c_rdata   = mm_rvalid ? mm_rdata : br_rdata;

    // ---------------------------------------------------- BRAM model
    reg [31:0] bram [0:BRAM_WORDS-1];
    reg [7:0]  ddr [0:DDR_BYTES-1];                          // DDR model (CPU and matmul DMA)
    reg        aw_seen = 0, w_seen = 0;
    reg [31:0] w_data;
    reg [3:0]  w_strb;
    integer    q;

    function in_bram(input [31:0] a);
        in_bram = a >= BRAM_BASE && a < BRAM_BASE + 4 * BRAM_WORDS;
    endfunction
    function in_cpu_ddr(input [31:0] a);                  // PicoRV32 -> S_AXI_HP0 -> DDR
        in_cpu_ddr = a >= DDR_BASE && a + 4 <= DDR_BASE + DDR_BYTES;
    endfunction

    always @(posedge clk) begin
        br_awready <= 0; br_wready <= 0; br_arready <= 0;
        if (br_bvalid && c_bready) br_bvalid <= 0;
        if (br_rvalid && c_rready) br_rvalid <= 0;

        if (c_awvalid && !aw_csr && !br_awready && !aw_seen) begin br_awready <= 1; aw_seen <= 1; end
        if (c_wvalid  && !aw_csr && !br_wready  && !w_seen)  begin br_wready  <= 1; w_seen  <= 1; w_data <= c_wdata; w_strb <= c_wstrb; end
        if (aw_seen && w_seen && !br_bvalid) begin
            if (in_cpu_ddr(c_awaddr)) begin
                for (q = 0; q < 4; q = q + 1)
                    if (w_strb[q]) ddr[c_awaddr - DDR_BASE + q] = w_data[8*q +: 8];
            end else if (!in_bram(c_awaddr)) begin
                $display("TB ERROR: CPU store to unmapped %08x", c_awaddr); errors = errors + 1;
            end else
                for (q = 0; q < 4; q = q + 1)
                    if (w_strb[q]) bram[(c_awaddr - BRAM_BASE) >> 2][8*q +: 8] <= w_data[8*q +: 8];
            br_bvalid <= 1; aw_seen <= 0; w_seen <= 0;
        end

        if (c_arvalid && !ar_csr && !br_arready && !br_rvalid) begin
            br_arready <= 1;
            if (in_cpu_ddr(c_araddr))
                br_rdata <= {ddr[c_araddr - DDR_BASE + 3], ddr[c_araddr - DDR_BASE + 2],
                             ddr[c_araddr - DDR_BASE + 1], ddr[c_araddr - DDR_BASE]};
            else if (!in_bram(c_araddr)) begin
                $display("TB ERROR: CPU load from unmapped %08x", c_araddr); errors = errors + 1;
                br_rdata <= 32'hXXXXXXXX;
            end else
                br_rdata <= bram[(c_araddr - BRAM_BASE) >> 2];
            br_rvalid <= 1;
        end
    end

    // ------------------------------------------------------ matmul_unit
    wire [31:0] m_araddr, m_awaddr;
    wire [7:0]  m_arlen, m_awlen, m_wstrb;
    wire [2:0]  m_arsize, m_awsize, m_arprot, m_awprot;
    wire [1:0]  m_arburst, m_awburst;
    wire [3:0]  m_arcache, m_awcache;
    wire        m_arvalid, m_rready, m_awvalid, m_wvalid, m_wlast, m_bready;
    wire [63:0] m_wdata;
    reg         m_arready = 0, m_rvalid = 0, m_rlast = 0, m_awready = 0, m_wready = 0, m_bvalid = 0;
    reg  [63:0] m_rdata = 0;
    wire        mm_irq, mm_nirq;

    localparam integer D = `SIM_D;
    sa_unit #(.D(D), .NPORTS(1)) mm (
        .aclk(clk), .aresetn(resetn),
        .s_axi_awaddr(c_awaddr[7:0]), .s_axi_awvalid(c_awvalid & aw_csr), .s_axi_awready(mm_awready),
        .s_axi_wdata(c_wdata), .s_axi_wstrb(c_wstrb), .s_axi_wvalid(c_wvalid & aw_csr), .s_axi_wready(mm_wready),
        .s_axi_bresp(mm_bresp), .s_axi_bvalid(mm_bvalid), .s_axi_bready(c_bready),
        .s_axi_araddr(c_araddr[7:0]), .s_axi_arvalid(c_arvalid & ar_csr), .s_axi_arready(mm_arready),
        .s_axi_rdata(mm_rdata), .s_axi_rresp(mm_rresp), .s_axi_rvalid(mm_rvalid), .s_axi_rready(c_rready),
        .m0_axi_araddr(m_araddr), .m0_axi_arlen(m_arlen), .m0_axi_arsize(m_arsize),
        .m0_axi_arburst(m_arburst), .m0_axi_arcache(m_arcache), .m0_axi_arprot(m_arprot),
        .m0_axi_arvalid(m_arvalid), .m0_axi_arready(m_arready),
        .m0_axi_rdata(m_rdata), .m0_axi_rresp(2'b00), .m0_axi_rlast(m_rlast),
        .m0_axi_rvalid(m_rvalid), .m0_axi_rready(m_rready),
        .m0_axi_awaddr(m_awaddr), .m0_axi_awlen(m_awlen), .m0_axi_awsize(m_awsize),
        .m0_axi_awburst(m_awburst), .m0_axi_awcache(m_awcache), .m0_axi_awprot(m_awprot),
        .m0_axi_awvalid(m_awvalid), .m0_axi_awready(m_awready),
        .m0_axi_wdata(m_wdata), .m0_axi_wstrb(m_wstrb), .m0_axi_wlast(m_wlast),
        .m0_axi_wvalid(m_wvalid), .m0_axi_wready(m_wready),
        .m0_axi_bresp(2'b00), .m0_axi_bvalid(m_bvalid), .m0_axi_bready(m_bready),
        .m1_axi_arready(1'b0), .m1_axi_rdata(64'd0), .m1_axi_rresp(2'd0), .m1_axi_rlast(1'b0),
        .m1_axi_rvalid(1'b0), .m1_axi_awready(1'b0), .m1_axi_wready(1'b0), .m1_axi_bresp(2'd0),
        .m1_axi_bvalid(1'b0),
        .m2_axi_arready(1'b0), .m2_axi_rdata(64'd0), .m2_axi_rresp(2'd0), .m2_axi_rlast(1'b0),
        .m2_axi_rvalid(1'b0), .m2_axi_awready(1'b0), .m2_axi_wready(1'b0), .m2_axi_bresp(2'd0),
        .m2_axi_bvalid(1'b0),
        .pcpi_valid(pcpi_valid), .pcpi_insn(pcpi_insn), .pcpi_rs1(pcpi_rs1), .pcpi_rs2(pcpi_rs2),
        .pcpi_wr(pcpi_wr), .pcpi_rd(pcpi_rd), .pcpi_wait(pcpi_wait), .pcpi_ready(pcpi_ready),
        .irq(mm_irq), .notify_irq(mm_nirq)
    );

    // --------------------------------------------- DDR model (matmul DMA)

    task ddr_check(input [31:0] addr, input integer bytes);
        if (addr < DDR_BASE || addr + bytes > DDR_BASE + DDR_BYTES) begin
            $display("TB ERROR: DMA outside DDR model: %08x", addr); errors = errors + 1;
        end
    endtask

    integer rb, wb, k;
    reg [31:0] r_addr, w_addr;
    reg [7:0]  r_len, w_len;
    initial begin : ddr_rd
        forever begin
            @(posedge clk);
            if (m_arvalid && m_arready) begin
                r_addr = m_araddr; r_len = m_arlen;
                ddr_check(r_addr, (r_len + 1) * 8);
                #1 m_arready = 0;
                for (rb = 0; rb <= r_len; rb = rb + 1) begin
                    repeat (rnd(6)) @(posedge clk);
                    #1;
                    for (k = 0; k < 8; k = k + 1) m_rdata[8*k +: 8] = ddr[r_addr - DDR_BASE + 8*rb + k];
                    m_rlast = rb == r_len; m_rvalid = 1;
                    @(posedge clk);
                    while (!m_rready) @(posedge clk);
                    #1 m_rvalid = 0; m_rlast = 0;
                end
            end else
                #1 m_arready = rnd(3) != 0;
        end
    end
    initial begin : ddr_wr
        forever begin
            @(posedge clk);
            if (m_awvalid && m_awready) begin
                w_addr = m_awaddr; w_len = m_awlen;
                ddr_check(w_addr, (w_len + 1) * 8);
                #1 m_awready = 0;
                for (wb = 0; wb <= w_len; wb = wb + 1) begin
                    repeat (rnd(6)) @(posedge clk);
                    #1 m_wready = 1;
                    @(posedge clk);
                    while (!m_wvalid) @(posedge clk);
                    for (k = 0; k < 8; k = k + 1) ddr[w_addr - DDR_BASE + 8*wb + k] = m_wdata[8*k +: 8];
                    #1 m_wready = 0;
                end
                repeat (rnd(6)) @(posedge clk);
                #1 m_bvalid = 1;
                @(posedge clk);
                while (!m_bready) @(posedge clk);
                #1 m_bvalid = 0;
            end else
                #1 m_awready = rnd(3) != 0;
        end
    end

    // --------------------------------------------------- "ARM" side
    reg [63:0] a_vec [0:NC*8-1];
    reg [63:0] b_vec [0:NC*8-1];
    reg [31:0] c_vec [0:NC*64-1];
    integer    i, e, cycles;
    reg [31:0] got;

    // VE arithmetic of one element (rtl/sysarray/sa_ve.v); automatic, and each
    // call in its own statement (xsim mixes up results of several calls of one
    // static function in a single expression)
    function automatic signed [63:0] clamp64(input signed [63:0] x, input signed [63:0] lo_, input signed [63:0] hi_);
        clamp64 = x < lo_ ? lo_ : x > hi_ ? hi_ : x;
    endfunction
    function automatic signed [31:0] ve_ref(input signed [31:0] a, input signed [31:0] b, input [7:0] op,
                                            input [1:0] ot, input integer scale, input integer shift,
                                            input integer zp, input integer lo, input integer hi);
        reg signed [63:0] r, q;
        begin
            case (op[2:0])
                3'd0: r = clamp64(a + b, -64'sd2147483648, 64'sd2147483647);
                3'd1: r = clamp64(a - b, -64'sd2147483648, 64'sd2147483647);
                3'd2: r = $signed(a[15:0]) * $signed(b[15:0]);
                3'd3: r = a > b ? a : b;
                3'd4: r = a < b ? a : b;
                default: r = a;
            endcase
            if (op[4] && r < 0) r = 0;
            q = op[5] ? ((r * $signed(scale[15:0]) + (shift == 0 ? 0 : (64'sd1 <<< (shift - 1)))) >>> shift) + zp : r;
            q = clamp64(q, lo, hi);
            ve_ref = clamp64(q, ot == 0 ? -128 : ot == 1 ? -32768 : -64'sd2147483648,
                                ot == 0 ?  127 : ot == 1 ?  32767 :  64'sd2147483647);
        end
    endfunction
    localparam integer I32MIN = 32'h80000000, I32MAX = 32'h7FFFFFFF;

    // ------------------------------------------------ performance counters
    // The firmware copies the counters to the perf area (program BRAM 0x1E00,
    // mailbox.h) and their number to MBOX_PERF_COUNT (0x8C). Indices: PC_* in
    // rtl/sysarray/sa_defs.vh.
    localparam integer PERF_W = 32'h1E00 / 4, MBOX_PERF_COUNT = 32'h8C / 4;
    localparam integer PC_CYCLES = 0, PC_CMD_LD = 1, PC_CMD_ST = 2, PC_CMD_EX = 3, PC_CMD_VE = 4,
                       PC_PCPI_QFULL = 5, PC_HAZ_LD = 7, PC_HAZ_ST = 8, PC_HAZ_EX = 9, PC_HAZ_VE = 10,
                       PC_STARVE = 12, PC_ALL_IDLE = 13, PC_EX_STEP = 14, PC_EX_USEFUL = 15,
                       PC_EX_TILES = 17, PC_LD_BUSY = 18, PC_LD_BEATS = 19, PC_ST_BUSY = 21,
                       PC_ST_BEATS = 22, PC_VE_ACTIVE = 24, PC_VE_GROUPS = 27;
    reg [31:0] pcv [0:31];
    integer    pci, perf_checks = 0;
    // x / total in per mille, printed as "12.3%"
    function integer pm(input [31:0] x, input [31:0] total); pm = total ? x * 1000 / total : 0; endfunction
    task perf_expect(input [31:0] got, input [31:0] want, input [8*24-1:0] what);
        begin
            perf_checks = perf_checks + 1;
            if (got !== want) begin
                errors = errors + 1;
                if (errors <= 10) $display("TB ERROR perf %0s = %0d, expected %0d", what, got, want);
            end
        end
    endtask
    // load the counters, check the count and the measurement window, print a breakdown
    task perf_load_report;
        reg [31:0] cyc, total;
        begin
            perf_expect(bram[MBOX + MBOX_PERF_COUNT], 32, "MBOX_PERF_COUNT");
            for (pci = 0; pci < 32; pci = pci + 1) pcv[pci] = bram[PERF_W + pci];
            cyc = pcv[PC_CYCLES]; total = bram[MBOX + 8];
            perf_checks = perf_checks + 1;
            // the counter window also covers the few instructions between mat_perf
            // and rdcycle on both ends (compiler-scheduled, tens of cycles)
            if (!(cyc >= total && cyc <= total + 128)) begin
                errors = errors + 1;
                $display("TB ERROR perf CYCLES = %0d outside the firmware window %0d .. +128", cyc, total);
            end
            $display("TB   perf: EX useful %0d.%0d%% fill %0d.%0d%% | head blocked LD %0d.%0d%% ST %0d.%0d%% EX %0d.%0d%% VE %0d.%0d%% | starve %0d.%0d%% idle %0d.%0d%% | CPU on full queue %0d.%0d%% | LD %0d B/%0d cyc, ST %0d B/%0d cyc",
                     pm(pcv[PC_EX_USEFUL], cyc) / 10, pm(pcv[PC_EX_USEFUL], cyc) % 10,
                     pm(pcv[PC_EX_STEP] - pcv[PC_EX_USEFUL], cyc) / 10, pm(pcv[PC_EX_STEP] - pcv[PC_EX_USEFUL], cyc) % 10,
                     pm(pcv[PC_HAZ_LD], cyc) / 10, pm(pcv[PC_HAZ_LD], cyc) % 10,
                     pm(pcv[PC_HAZ_ST], cyc) / 10, pm(pcv[PC_HAZ_ST], cyc) % 10,
                     pm(pcv[PC_HAZ_EX], cyc) / 10, pm(pcv[PC_HAZ_EX], cyc) % 10,
                     pm(pcv[PC_HAZ_VE], cyc) / 10, pm(pcv[PC_HAZ_VE], cyc) % 10,
                     pm(pcv[PC_STARVE], cyc) / 10, pm(pcv[PC_STARVE], cyc) % 10,
                     pm(pcv[PC_ALL_IDLE], cyc) / 10, pm(pcv[PC_ALL_IDLE], cyc) % 10,
                     pm(pcv[PC_PCPI_QFULL], cyc) / 10, pm(pcv[PC_PCPI_QFULL], cyc) % 10,
                     8 * pcv[PC_LD_BEATS], pcv[PC_LD_BUSY], 8 * pcv[PC_ST_BEATS], pcv[PC_ST_BUSY]);
        end
    endtask

`ifdef BW_TEST
    // -------------------------------------------------------- BW_TEST
    localparam [31:0] BW_SRC = DDR_BASE, BW_DST = DDR_BASE + 32'h10000;
    integer t;
    reg [31:0] bw_cyc;
    initial begin
        for (i = 0; i < DDR_BYTES; i = i + 1) ddr[i] = i < 65536 ? $random(seed) : 8'hA5;
        for (i = 0; i < BRAM_WORDS; i = i + 1) bram[i] = 0;
        $readmemh("fw.hex", bram);
        bram[MBOX + 2] = BW_SRC;
        bram[MBOX + 4] = BW_DST;
        repeat (20) @(posedge clk);
        resetn <= 1;
        cycles = 0;
        while (!trap && cycles < 4000000) begin @(posedge clk); cycles = cycles + 1; end
        repeat (5) @(posedge clk);
        if (!trap)                        begin $display("TB ERROR: no trap"); errors = errors + 1; end
        if (bram[MBOX] !== 32'h600D600D)  begin $display("TB ERROR: mailbox status %08x", bram[MBOX]); errors = errors + 1; end
        if (bram[MBOX + 6] !== 0)         begin $display("TB ERROR: fw errors, first %08x", bram[MBOX + 7]); errors = errors + 1; end
        for (t = 0; t < 3; t = t + 1)                    // three ST copies of the source
            for (i = 0; i < 65536; i = i + 1)
                if (ddr[BW_DST - DDR_BASE + 65536*t + i] !== ddr[i]) begin
                    errors = errors + 1;
                    if (errors <= 5) $display("TB ERROR: copy %0d byte %0d", t, i);
                end
        for (t = 0; t < 6; t = t + 1) begin
            bw_cyc = bram[MBOX + 17 + t];
            $display("TB BW test %0d: %0d cycles, %0d.%02d B/cycle", t, bw_cyc,
                     (t == 5 ? 131072 : 65536) / bw_cyc, ((t == 5 ? 131072 : 65536) * 100 / bw_cyc) % 100);
        end
        $display("TB %s: bandwidth test, %0d errors (simulated DDR model, not board numbers)",
                 errors ? "FAIL" : "PASS", errors);
        $finish;
    end
`elsif DESC_TEST
    // ------------------------------------------------------ DESC_TEST
    // Descriptor lists written into the DDR model like driver/pynq_matmul.py
    // DescList does; the desc_run firmware submits them.
    localparam [31:0] DA_ADDR = DDR_BASE, DB_ADDR = DDR_BASE + 32'h4000, DC_ADDR = DDR_BASE + 32'h8000,
                      DBIAS_ADDR = DDR_BASE + 32'h10000, DL_ADDR = DDR_BASE + 32'h40000;
    localparam integer SWD = 131072 / D, CWD = 262144 / (4 * D);   // SPAD / ACC words
    reg signed [7:0]  QA [0:63][0:127];
    reg signed [7:0]  QB [0:127][0:127];
    reg signed [31:0] QBIAS [0:63][0:127];
    integer di, dj, dk, runs = 0, dl_n;
    reg signed [31:0] dsum;
    task put64(input [31:0] a, input [63:0] v);
        integer j;
        for (j = 0; j < 8; j = j + 1) ddr[a - DDR_BASE + j] = v[8*j +: 8];
    endtask
    task d_put(input [7:0] op, input [7:0] fl, input [63:0] w1, input [63:0] w2, input [63:0] w3);
        begin
            put64(DL_ADDR + 64 * dl_n,      {dl_n[31:0], 16'd0, fl, op});
            put64(DL_ADDR + 64 * dl_n + 8,  w1);
            put64(DL_ADDR + 64 * dl_n + 16, w2);
            put64(DL_ADDR + 64 * dl_n + 24, w3);
            put64(DL_ADDR + 64 * dl_n + 32, 0); put64(DL_ADDR + 64 * dl_n + 40, 0);
            put64(DL_ADDR + 64 * dl_n + 48, 0); put64(DL_ADDR + 64 * dl_n + 56, 0);
            dl_n = dl_n + 1;
        end
    endtask
    function [31:0] la(input [3:0] m, input integer w); la = {m, 12'd0, w[15:0]}; endfunction
    task d_ld(input [31:0] ddr_a, input [31:0] laddr, input [15:0] rows, input [15:0] rb,
              input [31:0] pitch, input [1:0] mode, input [7:0] fl);
        d_put(8'h01, fl, {32'd0, ddr_a}, {rb, rows, laddr}, {30'd0, mode, pitch});
    endtask
    task d_st(input [31:0] ddr_a, input [31:0] laddr, input [15:0] rows, input [15:0] rb, input [31:0] pitch);
        d_put(8'h02, 8'd0, {32'd0, ddr_a}, {rb, rows, laddr}, {32'd0, pitch});
    endtask
    task d_ex(input [15:0] a, input [15:0] b, input [15:0] c, input [11:0] kt, input acc,
              input [11:0] rep, input [15:0] bstep, input [15:0] cstep, input [15:0] crow);
        d_put(8'h03, 8'd0, {3'd0, acc, kt, c, b, a}, {crow, cstep, bstep, 4'd0, rep}, 0);
    endtask

    // C = A x B (+ bias rows) as one list; reloc: A addressed as BASE0 + offset
    task run_desc_gemm(input integer M, input integer N, input integer K, input integer use_bias,
                       input integer reloc);
        integer t, bk;
        begin
            resetn <= 0;
            repeat (5) @(posedge clk);
            for (i = 0; i < DDR_BYTES; i = i + 1) ddr[i] = 8'hA5;
            for (di = 0; di < M; di = di + 1) for (dk = 0; dk < K; dk = dk + 1) begin
                QA[di][dk] = $random(seed); ddr[DA_ADDR - DDR_BASE + di*K + dk] = QA[di][dk];
            end
            for (dk = 0; dk < K; dk = dk + 1) for (dj = 0; dj < N; dj = dj + 1) begin
                QB[dk][dj] = $random(seed); ddr[DB_ADDR - DDR_BASE + dk*N + dj] = QB[dk][dj];
            end
            for (di = 0; di < M; di = di + 1) for (dj = 0; dj < N; dj = dj + 1) begin
                QBIAS[di][dj] = use_bias ? $random(seed) >>> 8 : 0;
                put64(DBIAS_ADDR + 4*(di*N + dj), {32'd0, QBIAS[di][dj]});   // (next word overwrites the top)
            end
            // the list: resident B, per strip {LD A, [LD bias], EX, ST}, END
            dl_n = 0;
            d_ld(DB_ADDR, la(2, 0), K, N, N, 1, 0);
            for (t = 0; t < M/D; t = t + 1) begin
                bk = t % 2;
                if (reloc) d_ld(t*D*K, la(1, bk * SWD/2), D, K, K, 1, 8'h01);      // BASE0 + offset
                else       d_ld(DA_ADDR + t*D*K, la(1, bk * SWD/2), D, K, K, 1, 0);
                if (use_bias) d_ld(DBIAS_ADDR + t*D*4*N, la(3, bk * CWD/2), D, 4*N, 4*N, 0, 0);
                d_ex(bk * SWD/2, 0, bk * CWD/2, K/D, use_bias, N/D, K, 1, N/D);
                d_st(DC_ADDR + t*D*4*N, la(3, bk * CWD/2), D, 4*N, 4*N);
            end
            d_put(8'h12, 8'd0, {32'd0, 32'hC0DE0000 + runs}, 0, 0);                 // END
            for (i = 0; i < BRAM_WORDS; i = i + 1) bram[i] = 0;
            $readmemh("fw.hex", bram);
            bram[MBOX + 32'h90/4] = DL_ADDR;
            bram[MBOX + 32'h94/4] = 0;
            bram[MBOX + 32'h98/4] = reloc ? DA_ADDR : 0;                          // BASE0
            repeat (5) @(posedge clk);
            resetn <= 1;
            cycles = 0;
            while (!trap && cycles < 3000000) begin @(posedge clk); cycles = cycles + 1; end
            repeat (5) @(posedge clk);
            if (!trap) begin $display("TB ERROR: desc GEMM no trap"); errors = errors + 1; end
            if (bram[MBOX] !== 32'h600D600D) begin $display("TB ERROR: mailbox status %08x", bram[MBOX]); errors = errors + 1; end
            if (bram[MBOX + 6] !== 0) begin $display("TB ERROR: desc ext status %08x", bram[MBOX + 16]); errors = errors + 1; end
            if (bram[MBOX + 32'hA8/4] !== 32'hC0DE0000 + runs) begin
                $display("TB ERROR: DL_STATUS %08x", bram[MBOX + 32'hA8/4]); errors = errors + 1;
            end
            if (bram[MBOX + 32'hAC/4] !== dl_n) begin
                $display("TB ERROR: DL_EXEC %0d, expected %0d", bram[MBOX + 32'hAC/4], dl_n); errors = errors + 1;
            end
            if (bram[MBOX + MBOX_PERF_COUNT] == 32) begin
                for (pci = 0; pci < 32; pci = pci + 1) pcv[pci] = bram[PERF_W + pci];
                perf_expect(pcv[28], dl_n, "FETCH_DESC");
                perf_expect(pcv[PC_EX_USEFUL] * D * D, M * N * K, "EX_USEFUL x D^2");
            end
            for (di = 0; di < M; di = di + 1) for (dj = 0; dj < N; dj = dj + 1) begin
                dsum = QBIAS[di][dj];
                for (dk = 0; dk < K; dk = dk + 1) dsum = dsum + QA[di][dk] * QB[dk][dj];
                e = DC_ADDR - DDR_BASE + 4*(di*N + dj);
                got = {ddr[e + 3], ddr[e + 2], ddr[e + 1], ddr[e]};
                if (got !== dsum) begin
                    errors = errors + 1;
                    if (errors <= 10) $display("TB ERROR desc GEMM %0dx%0dx%0d C[%0d][%0d] = %0d, expected %0d",
                                               M, N, K, di, dj, $signed(got), dsum);
                end
            end
            $display("TB desc GEMM %0dx%0dx%0d%0s%0s: %0d descriptors, %0d RISC-V cycles, %0d MAC/cycle",
                     M, N, K, use_bias ? " + bias" : "", reloc ? " (A relocated)" : "", dl_n,
                     bram[MBOX + 8], M*N*K / bram[MBOX + 8]);
            runs = runs + 1;
        end
    endtask

    // ---- co-simulation: lists built by the Python driver (gen_desc_cases.py)
    reg [7:0] dexp [0:32'h17FFF];
    `include "desc_cases.vh"
    integer dc, dj2, dbad;
    reg [31:0] dnd, doff, dnb;
    task run_py_case(input integer c);
        begin
            resetn <= 0;
            repeat (5) @(posedge clk);
            for (i = 0; i < DDR_BYTES; i = i + 1) ddr[i] = 8'hA5;
            load_dcase(c, dnd, doff, dnb);
            for (i = 0; i < BRAM_WORDS; i = i + 1) bram[i] = 0;
            $readmemh("fw.hex", bram);
            bram[MBOX + 32'h90/4] = DDR_BASE + 32'h40000;
            bram[MBOX + 32'h94/4] = 0;
            repeat (5) @(posedge clk);
            resetn <= 1;
            cycles = 0;
            while (!trap && cycles < 3000000) begin @(posedge clk); cycles = cycles + 1; end
            repeat (5) @(posedge clk);
            if (!trap || bram[MBOX] !== 32'h600D600D || bram[MBOX + 6] !== 0) begin
                $display("TB ERROR: py case %0d did not finish cleanly (status %08x, ext %08x)",
                         c, bram[MBOX], bram[MBOX + 16]);
                errors = errors + 1;
            end
            if (bram[MBOX + 32'hAC/4] !== dnd) begin
                $display("TB ERROR: py case %0d DL_EXEC %0d, expected %0d", c, bram[MBOX + 32'hAC/4], dnd);
                errors = errors + 1;
            end
            dbad = 0;
            for (dj2 = 0; dj2 < dnb; dj2 = dj2 + 1)
                if (ddr[doff + dj2] !== dexp[dj2]) begin
                    dbad = dbad + 1;
                    if (dbad <= 3) $display("TB ERROR: py case %0d out byte %0d = %02x, expected %02x",
                                            c, dj2, ddr[doff + dj2], dexp[dj2]);
                end
            for (dj2 = 0; dj2 < 8; dj2 = dj2 + 1)
                if (ddr[doff + dnb + dj2] !== 8'hA5) dbad = dbad + 1;          // nothing written past it
            errors = errors + dbad;
            $display("TB   -> %0d descriptors, %0d RISC-V cycles, %0s", dnd, bram[MBOX + 8], dbad ? "MISMATCH" : "ok");
            runs = runs + 1;
        end
    endtask

    initial begin
        run_desc_gemm(4*D, 4*D, 8*D, 0, 0);
        run_desc_gemm(3*D, 2*D, 5*D, 1, 0);
        run_desc_gemm(2*D, 6*D, 128, 0, 1);
        run_desc_gemm(D, D, D, 1, 1);
        for (dc = 0; dc < NDCASE; dc = dc + 1) run_py_case(dc);
        $display("TB %s: %0d descriptor-list runs (incl. Python-built lists), %0d errors, %0d performance-counter checks",
                 errors ? "FAIL" : "PASS", runs, errors, perf_checks);
        $finish;
    end
`elsif RING_TEST
    // ------------------------------------------------------- RING_TEST
    // The "ARM" writes 64-byte ring entries into DDR, rings the doorbell
    // (mailbox RING_TAIL) and consumes completion records in order; ring of
    // 4 entries, 9 submissions (wrap-around, at most 4 outstanding).
    localparam [31:0] RING = DDR_BASE + 32'h48000, CPL = DDR_BASE + 32'h49000, PBLK = DDR_BASE + 32'h4A000,
                      LISTS = DDR_BASE + 32'h4B000, SRC = DDR_BASE + 32'h10000, DST = DDR_BASE + 32'h20000;
    localparam [31:0] L_COPY = LISTS, L_PARAM = LISTS + 32'h400, L_BAD = LISTS + 32'h800, L_LOOP = LISTS + 32'hC00;
    localparam integer RSIZE = 4, NENT = 9;
    localparam [31:0] MB_RING_BASE = 32'hB0 / 4, MB_RING_SIZE = 32'hB4 / 4, MB_RING_TAIL = 32'hB8 / 4,
                      MB_RING_HEAD = 32'hBC / 4, MB_CPL_BASE = 32'hC0 / 4, MB_FW_STATE = 32'hC4 / 4,
                      MB_HEARTBEAT = 32'hCC / 4;
    localparam [7:0] T_RUN = 1, T_NOP = 2, T_RESET = 3, T_EXIT = 4;
    localparam [15:0] F_IRQ = 16'h100, F_PERF = 16'h200;
    integer nirq_edges = 0, runs = 0, ri, rj, rbad;
    reg     nirq_d = 0;
    initial begin                                            // a stuck ring must not hang the run
        #20_000_000;
        $display("TB FAIL: RING_TEST timeout (head %0d, tail %0d)", bram[MBOX + 32'hBC / 4], bram[MBOX + 32'hB8 / 4]);
        $finish;
    end
    always @(posedge clk) begin
        nirq_d <= mm_nirq;
        if (mm_nirq && !nirq_d) nirq_edges = nirq_edges + 1;
    end
    task put64(input [31:0] a, input [63:0] v);
        integer j;
        for (j = 0; j < 8; j = j + 1) ddr[a - DDR_BASE + j] = v[8*j +: 8];
    endtask
    function [31:0] get32(input [31:0] a);
        get32 = {ddr[a - DDR_BASE + 3], ddr[a - DDR_BASE + 2], ddr[a - DDR_BASE + 1], ddr[a - DDR_BASE]};
    endfunction
    // descriptor: header {tag, dyn slots [31:16], flags [15:8], opcode}, w1..w7
    task wdesc(input [31:0] a, input [7:0] op, input [15:0] dyn, input [7:0] fl, input [63:0] w1,
               input [63:0] w2, input [63:0] w3, input [63:0] w4);
        begin
            put64(a, {32'd0, dyn, fl, op}); put64(a + 8, w1); put64(a + 16, w2); put64(a + 24, w3);
            put64(a + 32, w4); put64(a + 40, 0); put64(a + 48, 0); put64(a + 56, 0);
        end
    endtask
    function [31:0] la(input [3:0] m, input integer w); la = {m, 12'd0, w[15:0]}; endfunction
    // ring entry i: type / flags, list, count, BASE0 / BASE1, parameter block
    reg [7:0]  e_type [0:NENT-1];
    reg [15:0] e_flags [0:NENT-1];
    reg [31:0] e_list [0:NENT-1], e_count [0:NENT-1], e_b0 [0:NENT-1], e_b1 [0:NENT-1], e_pb [0:NENT-1];
    task submit_entry(input integer i);
        reg [31:0] a;
        begin
            a = RING + 64 * (i % RSIZE);
            put64(a, {i[31:0] + 32'h5E0, 16'd0, e_flags[i] | e_type[i]});
            put64(a + 8, e_list[i]); put64(a + 16, e_count[i]); put64(a + 24, e_b0[i]); put64(a + 32, e_b1[i]);
            put64(a + 40, 0); put64(a + 48, 0); put64(a + 56, e_pb[i]);
            bram[MBOX + MB_RING_TAIL] = i + 1;                   // doorbell
        end
    endtask
    task check_cpl(input integer i, input [31:0] want_status_nz, input [31:0] want_exec, input [31:0] want_end);
        reg [31:0] a, st;
        begin
            a = CPL + 32 * (i % RSIZE);
            st = get32(a + 4);
            if (get32(a) !== i + 32'h5E0 || (want_status_nz ? st == 0 : st != 0) ||
                (want_exec != 32'hFFFFFFFF && get32(a + 12) !== want_exec) ||
                (want_end != 32'hFFFFFFFF && get32(a + 16) !== want_end)) begin
                $display("TB ERROR: completion %0d: seq %08x status %08x exec %0d end %08x", i,
                         get32(a), st, get32(a + 12), get32(a + 16));
                errors = errors + 1;
            end
            runs = runs + 1;
        end
    endtask

    initial begin
        for (ri = 0; ri < DDR_BYTES; ri = ri + 1) ddr[ri] = 8'hA5;
        for (ri = 0; ri < 8192; ri = ri + 1) ddr[SRC - DDR_BASE + ri] = $random(seed);
        // lists
        wdesc(L_COPY,       8'h01, 0, 8'h01, 0, {16'd256, 16'd1, la(1, 0)}, 64'd256, 0);       // LD BASE0 -> SPAD_A
        wdesc(L_COPY + 64,  8'h02, 0, 8'h03, 0, {16'd256, 16'd1, la(1, 0)}, 64'd256, 0);       // ST SPAD_A -> BASE1
        wdesc(L_COPY + 128, 8'h12, 0, 0, 64'h100, 0, 0, 0);                                     // END
        wdesc(L_PARAM,       8'h01, 16'h0001, 0, 0, {16'd256, 16'd1, la(2, 64)}, 64'd256, 0);   // LD ddr <- PARAM0
        wdesc(L_PARAM + 64,  8'h02, 16'h0011, 0, 0, {16'd256, 16'd1, la(2, 64)}, 64'd256, 0);   // ST ddr <- PARAM1
        wdesc(L_PARAM + 128, 8'h12, 0, 0, 64'h200, 0, 0, 0);
        wdesc(L_BAD, 8'h00, 0, 0, 0, 0, 0, 0);                                                  // opcode 0
        wdesc(L_LOOP,       8'h01, 16'h0081, 8'h01, 0, {16'd256, 16'd1, la(1, 128)}, 64'd256, 0); // LD BASE0 + PARAM0
        wdesc(L_LOOP + 64,  8'h02, 16'h0081, 8'h03, 0, {16'd256, 16'd1, la(1, 128)}, 64'd256, 0); // ST BASE1 + PARAM0
        wdesc(L_LOOP + 128, 8'h13, 16'h0021, 0, {32'd0, -32'sd2}, 64'd0, 64'd256, 0);           // LOOP_END, count <- PARAM2
        wdesc(L_LOOP + 192, 8'h12, 0, 0, 64'h300, 0, 0, 0);
        put64(PBLK,      {DST + 32'd512, SRC + 32'd512});                                       // P1, P0
        put64(PBLK + 8,  64'd0);
        put64(PBLK + 16, 64'd0);
        put64(PBLK + 24, 64'd0);
        put64(PBLK + 32, {32'd0, 32'd0});                                                       // loop block: P0 = 0
        put64(PBLK + 40, {32'd0, 32'd3});                                                       //   P2 = 3
        put64(PBLK + 48, 64'd0);
        put64(PBLK + 56, 64'd0);
        // entries
        for (ri = 0; ri < NENT; ri = ri + 1) begin
            e_flags[ri] = F_IRQ; e_count[ri] = 0; e_pb[ri] = 0; e_list[ri] = L_COPY;
            e_b0[ri] = SRC + 256 * ri; e_b1[ri] = DST + 256 * ri; e_type[ri] = T_RUN;
        end
        e_type[1] = T_NOP;
        e_list[2] = L_BAD;
        e_type[4] = T_RESET; e_flags[4] = 0;
        e_list[5] = L_PARAM; e_pb[5] = PBLK;
        e_list[6] = L_LOOP; e_pb[6] = PBLK + 32; e_b0[6] = SRC + 1024; e_b1[6] = DST + 1024; e_flags[6] = F_IRQ | F_PERF;
        e_count[7] = 1; e_flags[7] = 0; e_b0[7] = SRC + 2048; e_b1[7] = DST + 2048;       // only the LD runs
        e_type[8] = T_EXIT;
        // firmware + mailbox
        for (ri = 0; ri < BRAM_WORDS; ri = ri + 1) bram[ri] = 0;
        $readmemh("fw.hex", bram);
        bram[MBOX + MB_RING_BASE] = RING;
        bram[MBOX + MB_RING_SIZE] = RSIZE;
        bram[MBOX + MB_CPL_BASE]  = CPL;
        repeat (5) @(posedge clk);
        resetn <= 1;
        cycles = 0;
        while (bram[MBOX + MB_FW_STATE] !== 32'h52554E00 && cycles < 20000) begin @(posedge clk); cycles = cycles + 1; end
        if (bram[MBOX + MB_FW_STATE] !== 32'h52554E00) begin
            $display("TB ERROR: rt_fw not ready (FW_STATE %08x)", bram[MBOX + MB_FW_STATE]); errors = errors + 1;
        end
        // submit with at most RSIZE outstanding; check completions in order
        rj = 0;
        for (ri = 0; ri < NENT; ri = ri + 1) begin
            while (ri - bram[MBOX + MB_RING_HEAD] >= RSIZE) begin
                @(posedge clk);
                while (rj < bram[MBOX + MB_RING_HEAD]) begin      // consume finished entries first
                    case (rj)
                        0: check_cpl(0, 0, 3, 32'h100);
                        1: check_cpl(1, 0, 0, 32'hFFFFFFFF);
                        2: check_cpl(2, 1, 32'hFFFFFFFF, 32'hFFFFFFFF);
                        3: check_cpl(3, 0, 3, 32'h100);
                        4: check_cpl(4, 0, 0, 32'hFFFFFFFF);
                        default: ;
                    endcase
                    rj = rj + 1;
                end
            end
            submit_entry(ri);
            if (ri == 2) repeat (3000) @(posedge clk);         // let the ring drain once
        end
        cycles = 0;
        while (!trap && cycles < 400000) begin @(posedge clk); cycles = cycles + 1; end
        repeat (20) @(posedge clk);
        for (; rj < NENT; rj = rj + 1)
            case (rj)
                4: check_cpl(4, 0, 0, 32'hFFFFFFFF);
                5: check_cpl(5, 0, 3, 32'h200);
                6: check_cpl(6, 0, 1 + 3 * 3, 32'h300);        // LD, ST, LOOP_END x 3 + END
                7: check_cpl(7, 0, 1, 32'hFFFFFFFF);
                8: check_cpl(8, 0, 0, 32'hFFFFFFFF);
                default: ;
            endcase
        if (!trap || bram[MBOX] !== 32'h600D600D || bram[MBOX + MB_RING_HEAD] !== NENT) begin
            $display("TB ERROR: rt_fw did not exit cleanly (status %08x, head %0d)", bram[MBOX], bram[MBOX + MB_RING_HEAD]);
            errors = errors + 1;
        end
        // data: entries 0, 3 (copy), 5 (PARAM block), 6 (loop x 3), 7 (count 1: nothing stored)
        rbad = 0;
        for (ri = 0; ri < 256; ri = ri + 1) begin
            if (ddr[DST - DDR_BASE + ri] !== ddr[SRC - DDR_BASE + ri]) rbad = rbad + 1;
            if (ddr[DST - DDR_BASE + 768 + ri] !== ddr[SRC - DDR_BASE + 768 + ri]) rbad = rbad + 1;
            if (ddr[DST - DDR_BASE + 512 + ri] !== ddr[SRC - DDR_BASE + 512 + ri]) rbad = rbad + 1;
            if (ddr[DST - DDR_BASE + 256 + ri] !== 8'hA5) rbad = rbad + 1;       // NOP entry
            if (ddr[DST - DDR_BASE + 2048 + ri] !== 8'hA5) rbad = rbad + 1;      // count 1: LD only
        end
        for (ri = 0; ri < 768; ri = ri + 1)
            if (ddr[DST - DDR_BASE + 1024 + ri] !== ddr[SRC - DDR_BASE + 1024 + ri]) rbad = rbad + 1;
        if (ddr[DST - DDR_BASE + 1024 + 768] !== 8'hA5) rbad = rbad + 1;          // exactly 3 iterations
        if (rbad) begin
            $display("TB ERROR: ring data: %0d bytes wrong", rbad); errors = errors + rbad;
        end
        if (nirq_edges !== 7) begin
            $display("TB ERROR: %0d notify_irq edges, expected 7", nirq_edges); errors = errors + 1;
        end
        if (bram[MBOX + 32'h8C / 4] !== 32) begin
            $display("TB ERROR: PERF entry: MBOX_PERF_COUNT %0d", bram[MBOX + 32'h8C / 4]); errors = errors + 1;
        end
        $display("TB %s: rt_fw ring: %0d entries (ring of %0d), %0d completions checked, %0d notify edges, heartbeat %0d, %0d errors",
                 errors ? "FAIL" : "PASS", NENT, RSIZE, runs, nirq_edges, bram[MBOX + MB_HEARTBEAT], errors);
        $finish;
    end
`elsif VEC_TEST
    // ------------------------------------------------------- VEC_TEST
    localparam [31:0] V1_ADDR = DDR_BASE, V2_ADDR = DDR_BASE + 32'h10000, VD_ADDR = DDR_BASE + 32'h20000;
    integer runs = 0, vi, vj, vn2;
    reg signed [31:0] va, vb, vexp, vgot;

    function automatic integer esz(input [1:0] t); esz = t == 0 ? 1 : t == 1 ? 2 : 4; endfunction
    function automatic signed [31:0] get_el(input [31:0] addr, input [1:0] t, input integer i);
        integer o;
        begin
            o = addr - DDR_BASE + esz(t) * i;
            get_el = t == 0 ? $signed(ddr[o]) : t == 1 ? $signed({ddr[o + 1], ddr[o]})
                            : $signed({ddr[o + 3], ddr[o + 2], ddr[o + 1], ddr[o]});
        end
    endfunction

    task run_vec(input integer len, input [7:0] op, input [1:0] it, input [1:0] ot, input integer period,
                 input integer scale, input integer shift, input integer zp, input integer lo, input integer hi);
        begin
            resetn <= 0;
            repeat (5) @(posedge clk);
            for (i = 0; i < DDR_BYTES; i = i + 1) ddr[i] = 8'hA5;
            vn2 = period == 0 ? len : D * period;
            for (i = 0; i < len * esz(it); i = i + 1) ddr[V1_ADDR - DDR_BASE + i] = $random(seed);
            for (i = 0; i < vn2 * esz(it); i = i + 1) ddr[V2_ADDR - DDR_BASE + i] = $random(seed);
            if (it == 2)                                      // int32: moderate values, some saturation
                for (i = 0; i < len; i = i + 1) if (rnd(7) != 0) begin
                    e = V1_ADDR - DDR_BASE + 4*i; ddr[e + 3] = {8{ddr[e + 2][7]}};
                end
            for (i = 0; i < BRAM_WORDS; i = i + 1) bram[i] = 0;
            $readmemh("fw.hex", bram);
            bram[MBOX + 2]  = V1_ADDR;
            bram[MBOX + 3]  = V2_ADDR;
            bram[MBOX + 4]  = VD_ADDR;
            bram[MBOX + 24] = len;
            bram[MBOX + 25] = op;
            bram[MBOX + 26] = {ot, it};
            bram[MBOX + 27] = period;
            bram[MBOX + 28] = scale;
            bram[MBOX + 29] = shift;
            bram[MBOX + 30] = zp;
            bram[MBOX + 31] = lo;
            bram[MBOX + 32] = hi;

            repeat (5) @(posedge clk);
            resetn <= 1;
            cycles = 0;
            while (!trap && cycles < 3000000) begin @(posedge clk); cycles = cycles + 1; end
            repeat (5) @(posedge clk);

            if (!trap) begin $display("TB ERROR: vector run %0d no trap", runs); errors = errors + 1; end
            if (bram[MBOX] !== 32'h600D600D) begin $display("TB ERROR: mailbox status %08x", bram[MBOX]); errors = errors + 1; end
            if (bram[MBOX + 6] !== 0 || bram[MBOX + 16][1]) begin
                $display("TB ERROR: vector ext status %08x", bram[MBOX + 16]); errors = errors + 1;
            end
            for (vi = 0; vi < len; vi = vi + 1) begin
                va = get_el(V1_ADDR, it, vi);
                vb = get_el(V2_ADDR, it, period == 0 ? vi : vi % vn2);
                vexp = ve_ref(va, vb, op, ot, scale, shift, zp, lo, hi);
                vgot = get_el(VD_ADDR, ot, vi);
                if (vgot !== vexp) begin
                    errors = errors + 1;
                    if (errors <= 10) $display("TB ERROR vector run %0d element %0d = %0d, expected %0d (a %0d b %0d)",
                                               runs, vi, vgot, vexp, va, vb);
                end
            end
            for (vi = 0; vi < 8; vi = vi + 1)
                if (ddr[VD_ADDR - DDR_BASE + len * esz(ot) + vi] !== 8'hA5) begin
                    $display("TB ERROR: vector run %0d wrote past dst", runs); errors = errors + 1; vi = 8;
                end
            $display("TB vector op %0d%0s%0s %0s->%0s len %0d period %0d: %0d RISC-V cycles, %0d.%02d elements/cycle",
                     op[2:0], op[4] ? " relu" : "", op[5] ? " requant" : "",
                     it == 0 ? "i8" : it == 1 ? "i16" : "i32", ot == 0 ? "i8" : ot == 1 ? "i16" : "i32",
                     len, period, bram[MBOX + 8], len / bram[MBOX + 8], (len * 100 / bram[MBOX + 8]) % 100);
            // counters: VE groups / commands, DMA bytes (a periodic src2 is loaded into both banks)
            perf_load_report;
            perf_expect(pcv[PC_VE_GROUPS], len / D, "VE_GROUPS");
            perf_expect(pcv[PC_CMD_VE], bram[MBOX + 5], "CMD_VE (= chunks)");
            perf_expect(pcv[PC_EX_TILES], 0, "EX_TILES");
            perf_expect(8 * pcv[PC_ST_BEATS], len * esz(ot), "ST bytes");
            perf_expect(8 * pcv[PC_LD_BEATS], len * esz(it) +
                        (op[2:0] == 3'd5 ? 0 : period == 0 ? len * esz(it) : 2 * D * period * esz(it)), "LD bytes");
            runs = runs + 1;
        end
    endtask

    initial begin
        // lengths are multiples of 16 (both D); 20000 int8 = several chunks
        run_vec(20000, 8'h00, 0, 0, 0, 1, 0, 0, I32MIN, I32MAX);       // i8 + i8 -> i8 (saturating)
        run_vec(1040, 8'h02, 0, 1, 0, 1, 0, 0, I32MIN, I32MAX);        // i8 * i8 -> i16
        run_vec(2400, 8'h30, 2, 0, 4, 181, 15, -3, I32MIN, I32MAX);    // i32 + bias(period 4), relu, requant -> i8
        run_vec(784,  8'h03, 1, 1, 1, 1, 0, 0, -1000, 20000);           // max(i16, broadcast), clamp -> i16
        run_vec(4112, 8'h15, 2, 2, 0, 1, 0, 0, I32MIN, I32MAX);        // relu copy i32 -> i32 (D = 8: 514 groups)
        run_vec(5600, 8'h01, 0, 2, 3, 1, 0, 0, I32MIN, I32MAX);        // i8 - i8(period 3) -> i32
        run_vec(2048, 8'h22, 1, 2, 0, -300, 7, 1000, -500000, 500000); // i16 * i16, requant, clamp -> i32
        $display("TB %s: %0d vector operations, %0d errors, %0d performance-counter checks", errors ? "FAIL" : "PASS", runs, errors, perf_checks);
        $finish;
    end
`elsif GEMM_TEST
    // ------------------------------------------------------ GEMM_TEST
    localparam [31:0] GA_ADDR = DDR_BASE,           GB_ADDR = DDR_BASE + 32'h4000,
                      GC_ADDR = DDR_BASE + 32'h8000, GBIAS_ADDR = DDR_BASE + 32'h10000;
    reg signed [7:0]  GA [0:63][0:127];
    reg signed [7:0]  GB [0:127][0:127];
    reg signed [31:0] GBIAS [0:63][0:127];
    integer gi, gj, gk, runs = 0, total_macs, best_mpc;
    reg signed [31:0] gsum;

    task put32(input [31:0] addr, input [31:0] v);
        {ddr[addr - DDR_BASE + 3], ddr[addr - DDR_BASE + 2], ddr[addr - DDR_BASE + 1], ddr[addr - DDR_BASE]} = v;
    endtask

    reg signed [31:0] qexp;
    integer gemm_flags = 0;                          // MBOX_GEMM_FLAGS: 0 = tuned schedule
    task run_gemm(input integer M, input integer N, input integer K, input integer use_bias,
                  input integer quant, input integer relu, input integer scale, input integer shift,
                  input integer zp);
        begin
            resetn <= 0;
            repeat (5) @(posedge clk);
            for (i = 0; i < DDR_BYTES; i = i + 1) ddr[i] = 8'hA5;
            for (gi = 0; gi < M; gi = gi + 1) for (gk = 0; gk < K; gk = gk + 1) begin
                GA[gi][gk] = $random(seed); ddr[GA_ADDR - DDR_BASE + gi*K + gk] = GA[gi][gk];
            end
            for (gk = 0; gk < K; gk = gk + 1) for (gj = 0; gj < N; gj = gj + 1) begin
                GB[gk][gj] = $random(seed); ddr[GB_ADDR - DDR_BASE + gk*N + gj] = GB[gk][gj];
            end
            for (gi = 0; gi < M; gi = gi + 1) for (gj = 0; gj < N; gj = gj + 1) begin
                // int8 output: one bias vector of N (row 0), else a full M x N matrix
                GBIAS[gi][gj] = !use_bias ? 0 : quant && gi > 0 ? GBIAS[0][gj] : $random(seed) >>> (quant ? 14 : 8);
                if (!quant || gi == 0) put32(GBIAS_ADDR + 4*(gi*N + gj), GBIAS[gi][gj]);
            end
            for (i = 0; i < BRAM_WORDS; i = i + 1) bram[i] = 0;
            $readmemh("fw.hex", bram);
            bram[MBOX + 2]  = GA_ADDR;
            bram[MBOX + 3]  = GB_ADDR;
            bram[MBOX + 4]  = GC_ADDR;
            bram[MBOX + 12] = M;
            bram[MBOX + 13] = N;
            bram[MBOX + 14] = K;
            bram[MBOX + 15] = use_bias ? GBIAS_ADDR : 0;
            bram[MBOX + 33] = quant;                          // GEMM_Q
            bram[MBOX + 34] = gemm_flags;                     // GEMM_FLAGS
            bram[MBOX + 25] = (relu ? 32'h10 : 0) | 32'h20;   // V_OP: RELU, REQUANT
            bram[MBOX + 28] = scale;
            bram[MBOX + 29] = shift;
            bram[MBOX + 30] = zp;
            bram[MBOX + 31] = I32MIN;
            bram[MBOX + 32] = I32MAX;

            repeat (5) @(posedge clk);
            resetn <= 1;
            cycles = 0;
            while (!trap && cycles < 3000000) begin @(posedge clk); cycles = cycles + 1; end
            repeat (5) @(posedge clk);

            if (!trap) begin $display("TB ERROR: GEMM %0dx%0dx%0d no trap", M, N, K); errors = errors + 1; end
            if (bram[MBOX] !== 32'h600D600D) begin $display("TB ERROR: mailbox status %08x", bram[MBOX]); errors = errors + 1; end
            if (bram[MBOX + 6] !== 0 || bram[MBOX + 16][1]) begin
                $display("TB ERROR: GEMM ext status %08x", bram[MBOX + 16]); errors = errors + 1;
            end
            if (bram[MBOX + 5] !== (M/D) * (N/D)) begin $display("TB ERROR: tiles %0d", bram[MBOX + 5]); errors = errors + 1; end
            for (gi = 0; gi < M; gi = gi + 1) for (gj = 0; gj < N; gj = gj + 1) begin
                gsum = GBIAS[gi][gj];
                for (gk = 0; gk < K; gk = gk + 1) gsum = gsum + GA[gi][gk] * GB[gk][gj];
                if (quant) begin
                    qexp = ve_ref(gsum, 0, {2'b00, 1'b1, relu[0], 1'b0, 3'd5}, 2'd0, scale, shift, zp, I32MIN, I32MAX);
                    gsum = qexp;
                    got = {{24{ddr[GC_ADDR - DDR_BASE + gi*N + gj][7]}}, ddr[GC_ADDR - DDR_BASE + gi*N + gj]};
                end else begin
                    e = GC_ADDR - DDR_BASE + 4*(gi*N + gj);
                    got = {ddr[e + 3], ddr[e + 2], ddr[e + 1], ddr[e]};
                end
                if (got !== gsum) begin
                    errors = errors + 1;
                    if (errors <= 10) $display("TB ERROR GEMM %0dx%0dx%0d C[%0d][%0d] = %0d, expected %0d",
                                               M, N, K, gi, gj, $signed(got), gsum);
                end
            end
            if (ddr[GC_ADDR - DDR_BASE + (quant ? 1 : 4)*M*N] !== 8'hA5) begin
                $display("TB ERROR: byte after C overwritten"); errors = errors + 1;
            end
            $display("TB GEMM %0dx%0dx%0d%0s%0s%0s: %0d RISC-V cycles, %0d MAC/cycle, CPU CSR accesses %0d",
                     M, N, K, use_bias ? " + bias" : "", quant ? (relu ? " -> relu/requant int8" : " -> requant int8") : "",
                     gemm_flags == 3 ? " [M4 schedule]" : gemm_flags == 1 ? " [no prefetch]" : gemm_flags == 2 ? " [no B split]" :
                     gemm_flags == 4 ? " [B split forced]" : "",
                     bram[MBOX + 8], M*N*K / bram[MBOX + 8], csr_accesses);
            // counters: tiles, useful steps, DMA bytes, commands
            perf_load_report;
            perf_expect(pcv[PC_EX_TILES], (M/D) * (N/D), "EX_TILES");
            perf_expect(pcv[PC_EX_USEFUL], (M/D) * (N/D) * K, "EX_USEFUL");
            perf_expect(pcv[PC_EX_USEFUL] * D * D, M * N * K, "EX_USEFUL x D^2");
            perf_expect(8 * pcv[PC_ST_BEATS], (quant ? 1 : 4) * M * N, "ST bytes");
            perf_expect(8 * pcv[PC_LD_BEATS], K * N + M * K + (use_bias ? (quant ? 8 * N : 4 * M * N) : 0), "LD bytes");
            perf_expect(pcv[PC_CMD_ST], M / D, "CMD_ST");
            perf_expect(pcv[PC_CMD_VE], quant ? M / D : 0, "CMD_VE");
            if (quant) perf_expect(pcv[PC_VE_GROUPS], (M / D) * N, "VE_GROUPS");
            runs = runs + 1;
        end
    endtask

    initial begin
        // shapes in units of D (D = 8: 32x32x64, 24x16x40, 16x48x128, 8x8x8;
        // int8: 32x32x64, 24x48x40, 16x16x128)
        run_gemm(4*D, 4*D, 8*D, 0, 0, 0, 1, 0, 0);
        run_gemm(3*D, 2*D, 5*D, 1, 0, 0, 1, 0, 0);
        run_gemm(2*D, 6*D, 128, 1, 0, 0, 1, 0, 0);
        run_gemm(D, D, D, 0,       0, 0, 1, 0, 0);
        run_gemm(4*D, 4*D, 8*D, 1, 1, 1, 181, 15, -3);
        run_gemm(3*D, 6*D, 5*D, 0, 1, 0, -97, 12, 7);
        run_gemm(2*D, 2*D, 128, 1, 1, 1, 1, 0, 0);
        // the same GEMMs with parts of the schedule tuning switched off
        for (gemm_flags = 1; gemm_flags <= 3; gemm_flags = gemm_flags + 1) begin
            run_gemm(4*D, 4*D, 8*D, 0, 0, 0, 1, 0, 0);
            run_gemm(3*D, 2*D, 5*D, 1, 0, 0, 1, 0, 0);
            run_gemm(4*D, 4*D, 8*D, 1, 1, 1, 181, 15, -3);
        end
        // B split below the 16 KB threshold (the split path on small shapes)
        gemm_flags = 4;
        run_gemm(4*D, 4*D, 8*D, 0, 0, 0, 1, 0, 0);
        run_gemm(3*D, 2*D, 5*D, 1, 0, 0, 1, 0, 0);
        run_gemm(4*D, 4*D, 8*D, 1, 1, 1, 181, 15, -3);
        gemm_flags = 0;
        $display("TB %s: %0d GEMMs, %0d errors, %0d performance-counter checks", errors ? "FAIL" : "PASS", runs, errors, perf_checks);
        $finish;
    end
`else
    initial begin
        $readmemh("a.hex", a_vec);
        $readmemh("b.hex", b_vec);
        $readmemh("c.hex", c_vec);
        for (i = 0; i < BRAM_WORDS; i = i + 1) bram[i] = 0;
        $readmemh("fw.hex", bram);
        for (i = 0; i < DDR_BYTES; i = i + 1) ddr[i] = 8'hA5;

        // A/B rows: one 64-bit word per row; job i at +64*i
        for (i = 0; i < NC * 8; i = i + 1)
            for (k = 0; k < 8; k = k + 1) begin
                ddr[A_BASE - DDR_BASE + 8*i + k] = a_vec[i][8*k +: 8];
                ddr[B_BASE - DDR_BASE + 8*i + k] = b_vec[i][8*k +: 8];
            end

        bram[MBOX + 0] = 0;
        bram[MBOX + 1] = NC;
        bram[MBOX + 2] = A_BASE;
        bram[MBOX + 3] = B_BASE;
        bram[MBOX + 4] = C_BASE;
        bram[MBOX + 11] = DESC_BASE;

        // descriptors {A, B, DIM 8x8x8, 0} for the mat_trigger path
        for (i = 0; i < NC; i = i + 1)
            for (k = 0; k < 4; k = k + 1) begin
                got = k == 0 ? A_BASE + 64 * i : k == 1 ? B_BASE + 64 * i :
                      k == 2 ? {2'b0, 10'd8, 10'd8, 10'd8} : 32'd0;
                {ddr[DESC_BASE - DDR_BASE + 16*i + 4*k + 3], ddr[DESC_BASE - DDR_BASE + 16*i + 4*k + 2],
                 ddr[DESC_BASE - DDR_BASE + 16*i + 4*k + 1], ddr[DESC_BASE - DDR_BASE + 16*i + 4*k]} = got;
            end

        repeat (20) @(posedge clk);
        resetn <= 1;

        cycles = 0;
        while (!trap && cycles < 2000000) begin @(posedge clk); cycles = cycles + 1; end
        repeat (5) @(posedge clk);

        if (!trap)                          begin $display("TB ERROR: no trap"); errors = errors + 1; end
        if (bram[MBOX] !== 32'h600D600D)    begin $display("TB ERROR: mailbox status %08x", bram[MBOX]); errors = errors + 1; end
        // the CSR firmware reports the ID register; the PCPI firmware never reads CSRs
        if (bram[MBOX + 10] !== 32'h4D4D3038 && bram[MBOX + 10] !== 0) begin
            $display("TB ERROR: unit id %08x", bram[MBOX + 10]); errors = errors + 1;
        end
        if (bram[MBOX + 5] !== NC)          begin $display("TB ERROR: jobs done %0d / %0d", bram[MBOX + 5], NC); errors = errors + 1; end
        if (bram[MBOX + 6] !== 0)           begin $display("TB ERROR: fw errors %0d, first %08x", bram[MBOX + 6], bram[MBOX + 7]); errors = errors + 1; end

        for (i = 0; i < NC * 64; i = i + 1) begin
            e = C_BASE - DDR_BASE + 4 * i;
            got = {ddr[e + 3], ddr[e + 2], ddr[e + 1], ddr[e]};
            if (got !== c_vec[i]) begin
                errors = errors + 1;
                if (errors <= 10)
                    $display("TB ERROR job %0d C[%0d][%0d] = %0d, expected %0d", i / 64,
                             (i % 64) / 8, i % 8, $signed(got), $signed(c_vec[i]));
            end
        end
        if (ddr[C_BASE - DDR_BASE + NC * 256] !== 8'hA5) begin
            $display("TB ERROR: byte after last C overwritten"); errors = errors + 1;
        end

        $display("TB %s: %0d jobs, %0d errors; batch %0d cycles (%0d / job), accelerator %0d cycles (%0d / job), CPU CSR accesses %0d",
                 errors ? "FAIL" : "PASS", NC, errors, bram[MBOX + 8], bram[MBOX + 8] / NC,
                 bram[MBOX + 9], bram[MBOX + 9] / NC, csr_accesses);
        $finish;
    end
`endif
endmodule
