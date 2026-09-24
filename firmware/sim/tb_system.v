// System test: firmware (fw.hex) running on picorv32_axi drives the real
// sa_unit through its CSRs or its PCPI port, as wired in the overlay; the
// unit's DMA port 0 reads/writes a DDR model.
//   default   : 32 independent 8x8x8 jobs (firmware/matmul CSR path or
//               firmware/matmul_insn mat_trigger path)
//   GEMM_TEST : firmware/gemm, tiled GEMMs on the funct7 = 1 ISA
// Mirrors the overlay's RISC-V memory map:
//   0xC0000000  8 KB program BRAM (+ mailbox at 0xC0001F00)
//   0x80000000  matmul_unit CSRs
// and the matmul DMA view of DDR (identity-mapped physical addresses).
// The "ARM" side (initial block) plays driver/pynq_matmul.py.
`timescale 1ns / 1ps

`include "sa_macros.vh"
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
    reg        aw_seen = 0, w_seen = 0;
    reg [31:0] w_data;
    reg [3:0]  w_strb;
    integer    q;

    function in_bram(input [31:0] a);
        in_bram = a >= BRAM_BASE && a < BRAM_BASE + 4 * BRAM_WORDS;
    endfunction

    always @(posedge clk) begin
        br_awready <= 0; br_wready <= 0; br_arready <= 0;
        if (br_bvalid && c_bready) br_bvalid <= 0;
        if (br_rvalid && c_rready) br_rvalid <= 0;

        if (c_awvalid && !aw_csr && !br_awready && !aw_seen) begin br_awready <= 1; aw_seen <= 1; end
        if (c_wvalid  && !aw_csr && !br_wready  && !w_seen)  begin br_wready  <= 1; w_seen  <= 1; w_data <= c_wdata; w_strb <= c_wstrb; end
        if (aw_seen && w_seen && !br_bvalid) begin
            if (!in_bram(c_awaddr)) begin
                $display("TB ERROR: CPU store to unmapped %08x", c_awaddr); errors = errors + 1;
            end else
                for (q = 0; q < 4; q = q + 1)
                    if (w_strb[q]) bram[(c_awaddr - BRAM_BASE) >> 2][8*q +: 8] <= w_data[8*q +: 8];
            br_bvalid <= 1; aw_seen <= 0; w_seen <= 0;
        end

        if (c_arvalid && !ar_csr && !br_arready && !br_rvalid) begin
            br_arready <= 1;
            if (!in_bram(c_araddr)) begin
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
    wire        mm_irq;

    sa_unit #(.D(8), .NPORTS(1)) mm (
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
        .irq(mm_irq)
    );

    // --------------------------------------------- DDR model (matmul DMA)
    reg [7:0] ddr [0:DDR_BYTES-1];

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
`elsif GEMM_TEST
    // ------------------------------------------------------ GEMM_TEST
    localparam [31:0] GA_ADDR = DDR_BASE,           GB_ADDR = DDR_BASE + 32'h4000,
                      GC_ADDR = DDR_BASE + 32'h8000, GBIAS_ADDR = DDR_BASE + 32'h10000;
    reg signed [7:0]  GA [0:63][0:127];
    reg signed [7:0]  GB [0:127][0:63];
    reg signed [31:0] GBIAS [0:63][0:63];
    integer gi, gj, gk, runs = 0, total_macs, best_mpc;
    reg signed [31:0] gsum;

    task put32(input [31:0] addr, input [31:0] v);
        {ddr[addr - DDR_BASE + 3], ddr[addr - DDR_BASE + 2], ddr[addr - DDR_BASE + 1], ddr[addr - DDR_BASE]} = v;
    endtask

    task run_gemm(input integer M, input integer N, input integer K, input integer use_bias);
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
                GBIAS[gi][gj] = use_bias ? $random(seed) >>> 8 : 0;
                put32(GBIAS_ADDR + 4*(gi*N + gj), GBIAS[gi][gj]);
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
            if (bram[MBOX + 5] !== (M/8) * (N/8)) begin $display("TB ERROR: tiles %0d", bram[MBOX + 5]); errors = errors + 1; end
            for (gi = 0; gi < M; gi = gi + 1) for (gj = 0; gj < N; gj = gj + 1) begin
                gsum = GBIAS[gi][gj];
                for (gk = 0; gk < K; gk = gk + 1) gsum = gsum + GA[gi][gk] * GB[gk][gj];
                e = GC_ADDR - DDR_BASE + 4*(gi*N + gj);
                got = {ddr[e + 3], ddr[e + 2], ddr[e + 1], ddr[e]};
                if (got !== gsum) begin
                    errors = errors + 1;
                    if (errors <= 10) $display("TB ERROR GEMM %0dx%0dx%0d C[%0d][%0d] = %0d, expected %0d",
                                               M, N, K, gi, gj, $signed(got), gsum);
                end
            end
            if (ddr[GC_ADDR - DDR_BASE + 4*M*N] !== 8'hA5) begin
                $display("TB ERROR: byte after C overwritten"); errors = errors + 1;
            end
            $display("TB GEMM %0dx%0dx%0d%s: %0d RISC-V cycles, %0d MAC/cycle, CPU CSR accesses %0d",
                     M, N, K, use_bias ? " + bias" : "", bram[MBOX + 8], M*N*K / bram[MBOX + 8], csr_accesses);
            runs = runs + 1;
        end
    endtask

    initial begin
        run_gemm(32, 32, 64, 0);
        run_gemm(24, 16, 40, 1);
        run_gemm(16, 48, 128, 1);
        run_gemm(8, 8, 8, 0);
        $display("TB %s: %0d GEMMs, %0d errors", errors ? "FAIL" : "PASS", runs, errors);
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
