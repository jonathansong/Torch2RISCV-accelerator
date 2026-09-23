// Testbench for matmul_unit: golden vectors from gen_vectors.py (NumPy),
// an AXI4 memory model with random stalls, protocol checks and error
// injection, and an AXI4-Lite master for the CSRs.
`timescale 1ns / 1ps

module tb_matmul_unit;
    `include "n_cases.vh"

    localparam [31:0] MEM_BASE  = 32'h1000_0000;
    localparam integer MEM_BYTES = 65536;
    localparam [31:0] DIM_888   = {2'b0, 10'd8, 10'd8, 10'd8};

    localparam [7:0] CTRL = 8'h00, STATUS = 8'h04, SRC_A = 8'h08, SRC_B = 8'h0C,
                     DST = 8'h10, DIM = 8'h14, IRQ_STATUS = 8'h18, CYCLES = 8'h1C,
                     ID = 8'h20;

    reg aclk = 0, aresetn = 0;
    always #5 aclk = ~aclk;   // 100 MHz

    // ------------------------------------------------------------- DUT
    reg  [7:0]  s_awaddr = 0, s_araddr = 0;
    reg         s_awvalid = 0, s_wvalid = 0, s_bready = 0, s_arvalid = 0, s_rready = 0;
    reg  [31:0] s_wdata = 0;
    wire        s_awready, s_wready, s_bvalid, s_arready, s_rvalid;
    wire [1:0]  s_bresp, s_rresp;
    wire [31:0] s_rdata;

    wire [31:0] m_araddr, m_awaddr;
    wire [7:0]  m_arlen, m_awlen, m_wstrb;
    wire [2:0]  m_arsize, m_awsize, m_arprot, m_awprot;
    wire [1:0]  m_arburst, m_awburst;
    wire [3:0]  m_arcache, m_awcache;
    wire        m_arvalid, m_rready, m_awvalid, m_wvalid, m_wlast, m_bready;
    wire [63:0] m_wdata;
    reg         m_arready = 0, m_rvalid = 0, m_rlast = 0, m_awready = 0, m_wready = 0, m_bvalid = 0;
    reg  [63:0] m_rdata = 0;
    reg  [1:0]  m_rresp = 0, m_bresp = 0;
    wire        irq;

    matmul_unit dut (
        .aclk(aclk), .aresetn(aresetn),
        .s_axi_awaddr(s_awaddr), .s_axi_awvalid(s_awvalid), .s_axi_awready(s_awready),
        .s_axi_wdata(s_wdata), .s_axi_wstrb(4'hF), .s_axi_wvalid(s_wvalid), .s_axi_wready(s_wready),
        .s_axi_bresp(s_bresp), .s_axi_bvalid(s_bvalid), .s_axi_bready(s_bready),
        .s_axi_araddr(s_araddr), .s_axi_arvalid(s_arvalid), .s_axi_arready(s_arready),
        .s_axi_rdata(s_rdata), .s_axi_rresp(s_rresp), .s_axi_rvalid(s_rvalid), .s_axi_rready(s_rready),
        .m_axi_araddr(m_araddr), .m_axi_arlen(m_arlen), .m_axi_arsize(m_arsize),
        .m_axi_arburst(m_arburst), .m_axi_arcache(m_arcache), .m_axi_arprot(m_arprot),
        .m_axi_arvalid(m_arvalid), .m_axi_arready(m_arready),
        .m_axi_rdata(m_rdata), .m_axi_rresp(m_rresp), .m_axi_rlast(m_rlast),
        .m_axi_rvalid(m_rvalid), .m_axi_rready(m_rready),
        .m_axi_awaddr(m_awaddr), .m_axi_awlen(m_awlen), .m_axi_awsize(m_awsize),
        .m_axi_awburst(m_awburst), .m_axi_awcache(m_awcache), .m_axi_awprot(m_awprot),
        .m_axi_awvalid(m_awvalid), .m_axi_awready(m_awready),
        .m_axi_wdata(m_wdata), .m_axi_wstrb(m_wstrb), .m_axi_wlast(m_wlast),
        .m_axi_wvalid(m_wvalid), .m_axi_wready(m_wready),
        .m_axi_bresp(m_bresp), .m_axi_bvalid(m_bvalid), .m_axi_bready(m_bready),
        .irq(irq)
    );

    // ------------------------------------------------------ bookkeeping
    integer errors = 0;
    integer seed = 12345;
    integer rd_bursts = 0, wr_bursts = 0;
    reg     inject_rresp = 0, inject_bresp = 0;

    task fail(input [8*96-1:0] msg);
        begin
            errors = errors + 1;
            if (errors <= 20) $display("TB ERROR @%0t: %0s", $time, msg);
        end
    endtask

    function integer rnd(input integer max);   // 0..max
        begin
            rnd = $unsigned($random(seed)) % (max + 1);
        end
    endfunction

    // --------------------------------------------------- memory model
    reg [7:0] mem [0:MEM_BYTES-1];

    function in_mem(input [31:0] a, input integer len);
        in_mem = a >= MEM_BASE && a + len <= MEM_BASE + MEM_BYTES;
    endfunction

    task check_burst(input [31:0] addr, input [7:0] len, input [2:0] size, input [1:0] burst,
                     input is_write);
        begin
            if (size != 3'd3)  fail("burst size is not 8 bytes");
            if (burst != 2'b01) fail("burst type is not INCR");
            if (len > 8'd15)   fail("burst longer than 16 beats (not AXI3-safe)");
            if (addr[2:0] != 0) fail("unaligned burst address");
            if ((addr & 32'hFFF) + (len + 1) * 8 > 32'h1000) fail("burst crosses 4 KB boundary");
            if (!in_mem(addr, (len + 1) * 8)) fail("burst outside memory model");
        end
    endtask

    // read channel
    integer    rb;
    reg [31:0] r_addr;
    reg [7:0]  r_len;
    integer    k;
    initial begin : rd_slave
        forever begin
            @(posedge aclk);
            if (m_arvalid && m_arready) begin
                r_addr = m_araddr;
                r_len  = m_arlen;
                check_burst(r_addr, r_len, m_arsize, m_arburst, 0);
                rd_bursts = rd_bursts + 1;
                #1 m_arready = 0;
                for (rb = 0; rb <= r_len; rb = rb + 1) begin
                    repeat (rnd(3)) @(posedge aclk);
                    #1;
                    for (k = 0; k < 8; k = k + 1)
                        m_rdata[8*k +: 8] = in_mem(r_addr + 8*rb + k, 1) ? mem[r_addr + 8*rb + k - MEM_BASE] : 8'hXX;
                    m_rresp  = (inject_rresp && rb == 3) ? 2'b10 : 2'b00;
                    m_rlast  = rb == r_len;
                    m_rvalid = 1;
                    @(posedge aclk);
                    while (!m_rready) @(posedge aclk);
                    #1 m_rvalid = 0;
                    m_rlast = 0;
                end
            end else begin
                #1 m_arready = rnd(3) != 0;
            end
        end
    end

    // write channels
    integer    wb;
    reg [31:0] w_addr;
    reg [7:0]  w_len;
    initial begin : wr_slave
        forever begin
            @(posedge aclk);
            if (m_wvalid && !m_awvalid && w_len === 8'hxx)
                fail("W beat before any AW");
            if (m_awvalid && m_awready) begin
                w_addr = m_awaddr;
                w_len  = m_awlen;
                check_burst(w_addr, w_len, m_awsize, m_awburst, 1);
                wr_bursts = wr_bursts + 1;
                #1 m_awready = 0;
                for (wb = 0; wb <= w_len; wb = wb + 1) begin
                    repeat (rnd(3)) @(posedge aclk);
                    #1 m_wready = 1;
                    @(posedge aclk);
                    while (!m_wvalid) @(posedge aclk);
                    if (m_wstrb != 8'hFF)            fail("partial WSTRB");
                    if (m_wlast != (wb == w_len))    fail("WLAST on wrong beat");
                    for (k = 0; k < 8; k = k + 1)
                        mem[w_addr + 8*wb + k - MEM_BASE] = m_wdata[8*k +: 8];
                    #1 m_wready = 0;
                end
                repeat (rnd(4)) @(posedge aclk);
                #1 m_bresp = inject_bresp ? 2'b10 : 2'b00;
                m_bvalid = 1;
                @(posedge aclk);
                while (!m_bready) @(posedge aclk);
                #1 m_bvalid = 0;
            end else begin
                #1 m_awready = rnd(3) != 0;
            end
        end
    end

    // ------------------------------------------------ AXI4-Lite master
    reg aw_done, w_done;
    task csr_write(input [7:0] addr, input [31:0] data);
        begin
            @(posedge aclk);
            #1 s_awaddr = addr; s_awvalid = 1; s_wdata = data; s_wvalid = 1;
            aw_done = 0; w_done = 0;
            while (!(aw_done && w_done)) begin
                @(posedge aclk);
                if (s_awvalid && s_awready) aw_done = 1;
                if (s_wvalid && s_wready)   w_done  = 1;
                #1;
                if (aw_done) s_awvalid = 0;
                if (w_done)  s_wvalid  = 0;
            end
            s_bready = 1;
            @(posedge aclk);
            while (!s_bvalid) @(posedge aclk);
            if (s_bresp != 2'b00) fail("CSR write BRESP != OKAY");
            #1 s_bready = 0;
        end
    endtask

    task csr_read(input [7:0] addr, output [31:0] data);
        begin
            @(posedge aclk);
            #1 s_araddr = addr; s_arvalid = 1;
            @(posedge aclk);
            while (!s_arready) @(posedge aclk);
            #1 s_arvalid = 0; s_rready = 1;
            @(posedge aclk);
            while (!s_rvalid) @(posedge aclk);
            data = s_rdata;
            #1 s_rready = 0;
        end
    endtask

    task expect_csr(input [7:0] addr, input [31:0] want, input [8*48-1:0] what);
        reg [31:0] got;
        begin
            csr_read(addr, got);
            if (got !== want) begin
                errors = errors + 1;
                if (errors <= 20)
                    $display("TB ERROR @%0t: %0s = %08x, expected %08x", $time, what, got, want);
            end
        end
    endtask

    // ------------------------------------------------ test helpers
    reg [63:0] a_vec [0:NC*8-1];
    reg [63:0] b_vec [0:NC*8-1];
    reg [31:0] c_vec [0:NC*64-1];

    task load_matrix(input [31:0] addr, input integer vec_base, input is_b);
        integer r, j;
        reg [63:0] w;
        begin
            for (r = 0; r < 8; r = r + 1) begin
                w = is_b ? b_vec[vec_base + r] : a_vec[vec_base + r];
                for (j = 0; j < 8; j = j + 1)
                    mem[addr - MEM_BASE + 8*r + j] = w[8*j +: 8];
            end
        end
    endtask

    task poison(input [31:0] addr, input integer len);
        integer j;
        for (j = 0; j < len; j = j + 1) mem[addr - MEM_BASE + j] = 8'hA5;
    endtask

    function [31:0] mem32(input [31:0] addr);
        mem32 = {mem[addr - MEM_BASE + 3], mem[addr - MEM_BASE + 2],
                 mem[addr - MEM_BASE + 1], mem[addr - MEM_BASE]};
    endfunction

    reg [31:0] status;
    integer    timeout;
    task wait_done(input use_irq);
        begin
            timeout = 0;
            if (use_irq) begin
                while (!irq && timeout < 20000) begin @(posedge aclk); timeout = timeout + 1; end
                csr_read(STATUS, status);
            end else begin
                status = 0;
                while (!status[0] && timeout < 2000) begin
                    csr_read(STATUS, status);
                    timeout = timeout + 1;
                end
            end
            if (!status[0]) fail("timeout waiting for done");
        end
    endtask

    task run(input [31:0] a, input [31:0] b, input [31:0] c, input [2:0] ctrl_extra);
        begin
            csr_write(SRC_A, a);
            csr_write(SRC_B, b);
            csr_write(DST, c);
            csr_write(CTRL, {29'd0, ctrl_extra} | 32'h1);
        end
    endtask

    // check C of case `cs` at address c; guard bytes after it must be untouched
    integer e;
    task check_c(input integer cs, input [31:0] c);
        begin
            for (e = 0; e < 64; e = e + 1)
                if (mem32(c + 4*e) !== c_vec[64*cs + e]) begin
                    errors = errors + 1;
                    if (errors <= 20)
                        $display("TB ERROR case %0d: C[%0d][%0d] = %0d, expected %0d", cs,
                                 e / 8, e % 8, $signed(mem32(c + 4*e)), $signed(c_vec[64*cs + e]));
                end
            for (e = 256; e < 272; e = e + 1)
                if (mem[c - MEM_BASE + e] !== 8'hA5) fail("guard byte after C overwritten");
        end
    endtask

    // ---------------------------------------------------------- tests
    integer    cs, total_cycles, min_cycles, max_cycles, bursts_before;
    reg [31:0] a_addr, b_addr, c_addr, cyc;
    reg        use_irq;

    initial begin
        $readmemh("a.hex", a_vec);
        $readmemh("b.hex", b_vec);
        $readmemh("c.hex", c_vec);
        for (e = 0; e < MEM_BYTES; e = e + 1) mem[e] = 8'hA5;

        repeat (10) @(posedge aclk);
        #1 aresetn = 1;
        repeat (5) @(posedge aclk);

        // reset values
        expect_csr(ID, 32'h4D4D3038, "ID");
        expect_csr(DIM, DIM_888, "DIM reset value");
        expect_csr(STATUS, 32'h0, "STATUS after reset");
        expect_csr(IRQ_STATUS, 32'h0, "IRQ_STATUS after reset");

        // --- golden vectors
        total_cycles = 0; min_cycles = 32'h7FFFFFFF; max_cycles = 0;
        for (cs = 0; cs < NC; cs = cs + 1) begin
            // each case in its own 1 KB block, operands at varying 8-byte offsets
            a_addr = MEM_BASE + cs * 32'h400 + (cs % 8) * 8;
            b_addr = MEM_BASE + cs * 32'h400 + 32'h100 + ((cs * 3) % 8) * 8;
            c_addr = MEM_BASE + cs * 32'h400 + 32'h200 + ((cs * 5) % 8) * 8;
            load_matrix(a_addr, 8 * cs, 0);
            load_matrix(b_addr, 8 * cs, 1);
            poison(c_addr, 272);
            use_irq = cs % 2;
            run(a_addr, b_addr, c_addr, use_irq ? 3'b100 : 3'b000);
            wait_done(use_irq);
            if (status[2:1] != 2'b00) fail("error/busy set after a good run");
            check_c(cs, c_addr);
            csr_read(CYCLES, cyc);
            total_cycles = total_cycles + cyc;
            if (cyc < min_cycles) min_cycles = cyc;
            if (cyc > max_cycles) max_cycles = cyc;
            if (use_irq) begin
                if (!irq) fail("irq not asserted with irq_enable");
                csr_write(IRQ_STATUS, 1);                          // W1C
                @(posedge aclk);
                if (irq) fail("irq still asserted after IRQ_STATUS W1C");
                expect_csr(IRQ_STATUS, 0, "IRQ_STATUS after W1C");
                csr_write(CTRL, 0);                                 // irq_enable off
            end else begin
                if (irq) fail("irq asserted without irq_enable");
                expect_csr(IRQ_STATUS, 1, "IRQ_STATUS latched without irq_enable");
                csr_write(IRQ_STATUS, 1);   // else irq fires as soon as it is enabled
            end
        end

        // --- config writes are ignored while busy; start while busy is ignored
        cs = 5;
        a_addr = MEM_BASE + 32'h8000; b_addr = MEM_BASE + 32'h8100; c_addr = MEM_BASE + 32'h8200;
        load_matrix(a_addr, 8 * cs, 0);
        load_matrix(b_addr, 8 * cs, 1);
        poison(c_addr, 272);
        run(a_addr, b_addr, c_addr, 3'b000);
        csr_write(SRC_A, MEM_BASE + 32'h9000);
        csr_write(DST, MEM_BASE + 32'h9200);
        csr_write(CTRL, 1);
        wait_done(0);
        check_c(cs, c_addr);
        expect_csr(SRC_A, a_addr, "SRC_A written while busy");

        // --- soft reset clears status and IRQ_STATUS
        csr_write(CTRL, 32'h2);
        expect_csr(STATUS, 0, "STATUS after soft reset");
        expect_csr(IRQ_STATUS, 0, "IRQ_STATUS after soft reset");

        // --- error: DIM not 8x8x8 (no bus traffic)
        bursts_before = rd_bursts + wr_bursts;
        csr_write(DIM, {2'b0, 10'd4, 10'd4, 10'd4});
        csr_write(CTRL, 1);
        wait_done(0);
        if (status !== {20'd0, 4'd1, 5'd0, 3'b101}) fail("DIM error: wrong STATUS");
        if (rd_bursts + wr_bursts != bursts_before) fail("DIM error: unit touched memory");
        csr_write(DIM, DIM_888);

        // --- error: misaligned A
        run(MEM_BASE + 32'h8004, b_addr, c_addr, 3'b000);
        wait_done(0);
        if (status !== {20'd0, 4'd2, 5'd0, 3'b101}) fail("misaligned A: wrong STATUS");

        // --- error: C would cross a 4 KB page
        run(a_addr, b_addr, MEM_BASE + 32'hAF08, 3'b000);
        wait_done(0);
        if (status !== {20'd0, 4'd2, 5'd0, 3'b101}) fail("4 KB crossing: wrong STATUS");
        if (rd_bursts + wr_bursts != bursts_before) fail("address error: unit touched memory");

        // --- error: SLVERR while reading; C must not be written
        poison(c_addr, 272);
        inject_rresp = 1;
        run(a_addr, b_addr, c_addr, 3'b000);
        wait_done(0);
        inject_rresp = 0;
        if (status !== {20'd0, 4'd3, 5'd0, 3'b101}) fail("read SLVERR: wrong STATUS");
        if (mem32(c_addr) !== 32'hA5A5A5A5) fail("read SLVERR: C was written anyway");

        // --- error: SLVERR on write response
        inject_bresp = 1;
        run(a_addr, b_addr, c_addr, 3'b000);
        wait_done(0);
        inject_bresp = 0;
        if (status !== {20'd0, 4'd4, 5'd0, 3'b101}) fail("write SLVERR: wrong STATUS");

        // --- recovery after errors
        poison(c_addr, 272);
        run(a_addr, b_addr, c_addr, 3'b000);
        wait_done(0);
        if (status !== 32'h1) fail("recovery run: wrong STATUS");
        check_c(cs, c_addr);

        repeat (20) @(posedge aclk);
        $display("TB %s: %0d golden cases + CSR/error tests, %0d errors; cycles per 8x8x8 matmul: min %0d / avg %0d / max %0d (AXI stalls randomized)",
                 errors ? "FAIL" : "PASS", NC, errors, min_cycles, total_cycles / NC, max_cycles);
        $finish;
    end

    initial begin
        #5_000_000;
        $display("TB FAIL: global timeout");
        $finish;
    end
endmodule
