// Full-unit test of sa_unit (double-buffered accelerator, M1).
//  1. Phase 2-4 compatibility: the tb_matmul_unit tests (NumPy golden vectors
//     through the CSR and mat_trigger paths, errors, irq, busy writes,
//     PCPI legacy instructions), ported to the new top level.
//  2. New ISA (funct7 = 1): double-buffered tiled GEMMs incl. bias preload,
//     checked against a behavioral golden model.
//  3. Scoreboard: a random command stream with frequent bank conflicts; the
//     final local memories and DDR must equal a sequential reference model.
//  4. New-ISA errors (sticky error, dropped commands, mat_reset), CAPS/EXT.
`timescale 1ns / 1ps

`include "sa_macros.vh"
module tb_sa_unit;
    `include "n_cases.vh"
    parameter  integer D  = 8;                    // make sim TB=tb_sa_unit GEN=D=16
    parameter  integer PERF = 1;                  // GEN="PERF=0": build without counters
    localparam integer AW = 32 * D;               // widest local word (ACC)

    localparam [31:0] MEM_BASE  = 32'h1000_0000;
    localparam integer MEM_BYTES = 262144;
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

    reg         pcpi_valid = 0;
    reg  [31:0] pcpi_insn = 0, pcpi_rs1 = 0, pcpi_rs2 = 0;
    wire        pcpi_wr, pcpi_wait, pcpi_ready;
    wire [31:0] pcpi_rd;

    sa_unit #(.D(D), .NPORTS(1), .PERF(PERF)) dut (
        .aclk(aclk), .aresetn(aresetn),
        .s_axi_awaddr(s_awaddr), .s_axi_awvalid(s_awvalid), .s_axi_awready(s_awready),
        .s_axi_wdata(s_wdata), .s_axi_wstrb(4'hF), .s_axi_wvalid(s_wvalid), .s_axi_wready(s_wready),
        .s_axi_bresp(s_bresp), .s_axi_bvalid(s_bvalid), .s_axi_bready(s_bready),
        .s_axi_araddr(s_araddr), .s_axi_arvalid(s_arvalid), .s_axi_arready(s_arready),
        .s_axi_rdata(s_rdata), .s_axi_rresp(s_rresp), .s_axi_rvalid(s_rvalid), .s_axi_rready(s_rready),
        .m0_axi_araddr(m_araddr), .m0_axi_arlen(m_arlen), .m0_axi_arsize(m_arsize),
        .m0_axi_arburst(m_arburst), .m0_axi_arcache(m_arcache), .m0_axi_arprot(m_arprot),
        .m0_axi_arvalid(m_arvalid), .m0_axi_arready(m_arready),
        .m0_axi_rdata(m_rdata), .m0_axi_rresp(m_rresp), .m0_axi_rlast(m_rlast),
        .m0_axi_rvalid(m_rvalid), .m0_axi_rready(m_rready),
        .m0_axi_awaddr(m_awaddr), .m0_axi_awlen(m_awlen), .m0_axi_awsize(m_awsize),
        .m0_axi_awburst(m_awburst), .m0_axi_awcache(m_awcache), .m0_axi_awprot(m_awprot),
        .m0_axi_awvalid(m_awvalid), .m0_axi_awready(m_awready),
        .m0_axi_wdata(m_wdata), .m0_axi_wstrb(m_wstrb), .m0_axi_wlast(m_wlast),
        .m0_axi_wvalid(m_wvalid), .m0_axi_wready(m_wready),
        .m0_axi_bresp(m_bresp), .m0_axi_bvalid(m_bvalid), .m0_axi_bready(m_bready),

        // ports 1 and 2 unused in M1
        .m1_axi_arready(1'b0), .m1_axi_rdata(64'd0), .m1_axi_rresp(2'd0), .m1_axi_rlast(1'b0),
        .m1_axi_rvalid(1'b0), .m1_axi_awready(1'b0), .m1_axi_wready(1'b0), .m1_axi_bresp(2'd0),
        .m1_axi_bvalid(1'b0),
        .m2_axi_arready(1'b0), .m2_axi_rdata(64'd0), .m2_axi_rresp(2'd0), .m2_axi_rlast(1'b0),
        .m2_axi_rvalid(1'b0), .m2_axi_awready(1'b0), .m2_axi_wready(1'b0), .m2_axi_bresp(2'd0),
        .m2_axi_bvalid(1'b0),
        .pcpi_valid(pcpi_valid), .pcpi_insn(pcpi_insn), .pcpi_rs1(pcpi_rs1), .pcpi_rs2(pcpi_rs2),
        .pcpi_wr(pcpi_wr), .pcpi_rd(pcpi_rd), .pcpi_wait(pcpi_wait), .pcpi_ready(pcpi_ready),
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
                    // beat 3, or the last beat of a shorter burst (D = 16: 1-beat rows)
                    m_rresp  = (inject_rresp && rb == (r_len < 3 ? r_len : 3)) ? 2'b10 : 2'b00;
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

    // ------------------------------------------ PCPI master (PicoRV32 model)
    localparam [2:0] F_TRIGGER = 0, F_STATUS = 1, F_RESET = 2, F_WAIT = 3, F_CYCLES = 4;

    function [31:0] custom0(input [2:0] f3, input [6:0] f7);
        custom0 = {f7, 5'd11, 5'd10, f3, 5'd5, 7'b0001011};   // rd=x5 rs1=x10 rs2=x11
    endfunction

    // Issue one instruction like picorv32 does: hold valid/insn/rs until
    // ready; the core would trap if pcpi_wait were low for 16 cycles.
    integer    pc_cycles, pc_nowait;
    reg [31:0] pc_rd;
    reg        pc_wr;
    task pcpi_exec(input [31:0] insn, input [31:0] rs1, input [31:0] rs2);
        begin
            @(posedge aclk);
            #1 pcpi_valid = 1; pcpi_insn = insn; pcpi_rs1 = rs1; pcpi_rs2 = rs2;
            pc_cycles = 0; pc_nowait = 0;
            @(posedge aclk);
            while (!pcpi_ready && pc_nowait < 16) begin
                pc_nowait = pcpi_wait ? 0 : pc_nowait + 1;
                pc_cycles = pc_cycles + 1;
                @(posedge aclk);
            end
            pc_rd = pcpi_rd;
            pc_wr = pcpi_wr;
            if (!pcpi_ready) pc_cycles = -1;       // would have trapped
            #1 pcpi_valid = 0; pcpi_insn = 0;
        end
    endtask

    task mat_op(input [2:0] f3, input [31:0] rs1, input [31:0] rs2);
        begin
            pcpi_exec(custom0(f3, 7'd0), rs1, rs2);
            if (pc_cycles < 0) fail("PCPI instruction timed out (would trap)");
        end
    endtask

    // descriptor {A, B, DIM, 0} at d
    task put_desc(input [31:0] d, input [31:0] a, input [31:0] b, input [31:0] dimv);
        integer j;
        reg [127:0] w;
        begin
            w = {32'd0, dimv, b, a};
            for (j = 0; j < 16; j = j + 1) mem[d - MEM_BASE + j] = w[8*j +: 8];
        end
    endtask


    // ==================================================== new ISA helpers
    localparam [7:0] V_OP = 0, V_LEN = 1, V_DST = 2, V_TYPES = 3, V_MOD = 4, V_SCALE = 5,
                     V_SHIFT = 6, V_ZP = 7, V_LO = 8, V_HI = 9;
    localparam [2:0] VADD = 0, VSUB = 1, VMUL = 2, VMAX = 3, VMIN = 4, VCOPY = 5;
    localparam [1:0] T8 = 0, T16 = 1, T32 = 2;
    localparam [7:0] K_LD_ROWS = 0, K_LD_RB = 1, K_LD_PITCH = 2, K_LD_MODE = 3,
                     K_ST_ROWS = 4, K_ST_RB = 5, K_ST_PITCH = 6,
                     K_EX_REP = 7, K_EX_BSTEP = 8, K_EX_CSTEP = 9, K_EX_CROW = 10;
    localparam [3:0] M_A = 1, M_B = 2, M_C = 3;
    localparam integer SW = 131072 / D, CW = 262144 / (4 * D);   // SPAD / ACC words

    task n_op(input [2:0] f3, input [31:0] rs1, input [31:0] rs2);
        begin
            pcpi_exec(custom0(f3, 7'd1), rs1, rs2);
            if (pc_cycles < 0) fail("new-ISA instruction timed out (would trap)");
        end
    endtask
    task cfg(input [7:0] key, input [31:0] v); n_op(3'd0, key, v); endtask
    task ld(input [31:0] ddr, input [3:0] m, input [15:0] w); n_op(3'd1, ddr, {m, 12'd0, w}); endtask
    task st(input [31:0] ddr, input [3:0] m, input [15:0] w); n_op(3'd2, ddr, {m, 12'd0, w}); endtask
    task ex(input [15:0] a, input [15:0] b, input [15:0] c, input [11:0] kt, input acc);
        n_op(3'd3, {b, a}, {3'd0, acc, kt, c});
    endtask
    reg [31:0] xst;
    task fence(input [3:0] mask); begin n_op(3'd4, mask, 0); xst = pc_rd; end endtask
    task cfg_ld(input integer rows, input integer rb, input integer pitch, input integer mode);
        begin cfg(K_LD_ROWS, rows); cfg(K_LD_RB, rb); cfg(K_LD_PITCH, pitch); cfg(K_LD_MODE, mode); end
    endtask
    task cfg_st(input integer rows, input integer rb, input integer pitch);
        begin cfg(K_ST_ROWS, rows); cfg(K_ST_RB, rb); cfg(K_ST_PITCH, pitch); end
    endtask
    task cfg_ex(input integer r, input integer bstep, input integer cstep, input integer crow);
        begin cfg(K_EX_REP, r); cfg(K_EX_BSTEP, bstep); cfg(K_EX_CSTEP, cstep); cfg(K_EX_CROW, crow); end
    endtask

    // direct views of the unit's memories (bank = upper half)
    reg [AW-1:0] cmp_h, cmp_r;
    function [8*D-1:0] hw_a(input integer w); hw_a = w < SW/2 ? dut.spad_a.bank[0].ram.ram_block[w] : dut.spad_a.bank[1].ram.ram_block[w - SW/2]; endfunction
    function [8*D-1:0] hw_b(input integer w); hw_b = w < SW/2 ? dut.spad_b.bank[0].ram.ram_block[w] : dut.spad_b.bank[1].ram.ram_block[w - SW/2]; endfunction
    function [AW-1:0]  hw_c(input integer w); hw_c = w < CW/2 ? dut.accm.bank[0].ram.ram_block[w]   : dut.accm.bank[1].ram.ram_block[w - CW/2];   endfunction

    // --------------------------------------------------- tiled GEMM test
    reg signed [7:0]  GA [0:127][0:127];
    reg signed [7:0]  GB [0:127][0:127];
    reg signed [31:0] GBIAS [0:127][0:127];
    integer gi, gj, gk, gt, gbk, gti, gtj;
    reg signed [31:0] gsum, ggot;
    integer gemm_cycles, gemm_cycles_main;
    task gemm(input integer M, input integer N, input integer K, input [31:0] aa, input [31:0] ba,
              input [31:0] ca, input integer use_bias, input [31:0] biasa);
        integer t0;
        begin
            for (gi = 0; gi < M; gi = gi + 1) for (gk = 0; gk < K; gk = gk + 1) begin
                GA[gi][gk] = $random(seed); mem[aa - MEM_BASE + gi*K + gk] = GA[gi][gk];
            end
            for (gk = 0; gk < K; gk = gk + 1) for (gj = 0; gj < N; gj = gj + 1) begin
                GB[gk][gj] = $random(seed); mem[ba - MEM_BASE + gk*N + gj] = GB[gk][gj];
            end
            for (gi = 0; gi < M; gi = gi + 1) for (gj = 0; gj < N; gj = gj + 1) begin
                GBIAS[gi][gj] = use_bias ? $random(seed) >>> 8 : 0;
                {mem[biasa - MEM_BASE + 4*(gi*N + gj) + 3], mem[biasa - MEM_BASE + 4*(gi*N + gj) + 2],
                 mem[biasa - MEM_BASE + 4*(gi*N + gj) + 1], mem[biasa - MEM_BASE + 4*(gi*N + gj)]} = GBIAS[gi][gj];
            end
            poison(ca, 4*M*N + 16);
            t0 = $time;
            // B resident: one INTERLEAVE load of the whole K x N matrix puts
            // column strip jt at SPAD_B word jt*K (word k = B[k][D*jt .. D*jt+D-1])
            cfg_ld(K, N, N, 1);
            ld(ba, M_B, 0);
            // M2: one mat_exec per A strip computes its N/D C tiles, laid out
            // row-major in ACC (tile j row i at word j + i*N/D), so one store
            // writes the whole D x N strip of C
            cfg_ex(N/D, K, 1, N/D);
            cfg_st(D, 4*N, 4*N);
            for (gti = 0; gti < M/D; gti = gti + 1) begin
                gbk = gti % 2;                                             // strips alternate banks
                cfg_ld(D, K, K, 1);
                ld(aa + gti*D*K, M_A, gbk * SW/2);
                if (use_bias) begin
                    cfg_ld(D, 4*N, 4*N, 0);                                // bias strip, same layout as C
                    ld(biasa + gti*D*4*N, M_C, gbk * CW/2);
                end
                ex(gbk * SW/2, 0, gbk * CW/2, K/D, use_bias);
                st(ca + gti*D*4*N, M_C, gbk * CW/2);
            end
            fence(0);
            gemm_cycles = ($time - t0) / 10;
            cfg_ex(1, 0, 0, 1);                                            // back to single tiles
            if (xst[1]) fail("GEMM: sticky error");
            for (gi = 0; gi < M; gi = gi + 1) for (gj = 0; gj < N; gj = gj + 1) begin
                gsum = GBIAS[gi][gj];
                for (gk = 0; gk < K; gk = gk + 1) gsum = gsum + GA[gi][gk] * GB[gk][gj];
                ggot = mem32(ca + 4*(gi*N + gj));
                if (ggot !== gsum) begin
                    errors = errors + 1;
                    if (errors <= 20) $display("TB ERROR GEMM %0dx%0dx%0d C[%0d][%0d] = %0d, expected %0d",
                                               M, N, K, gi, gj, ggot, gsum);
                end
            end
            for (gi = 4*M*N; gi < 4*M*N + 16; gi = gi + 1)
                if (mem[ca - MEM_BASE + gi] !== 8'hA5) fail("GEMM: guard after C overwritten");
        end
    endtask

    // ------------------------------------------ random stream vs reference
    reg [8*D-1:0] sh_a [0:SW-1];
    reg [8*D-1:0] sh_b [0:SW-1];
    reg [AW-1:0]  sh_c [0:CW-1];
    localparam [31:0] RS = MEM_BASE + 32'h30000, RD = MEM_BASE + 32'h38000;   // 32 KB each
    reg [7:0] rd_sh [0:32767];
    integer   ri, rj;
    reg [8*D-1:0] rk;
    reg [31:0] ext_rd;

    task ref_ld(input [31:0] d, input [3:0] m, input integer w, input integer rows, input integer rb,
                input integer pitch, input integer mode);
        integer rr, bb, wd, byt, wb_, wpr;
        begin
            wb_ = m == M_C ? 4*D : D;
            wpr = (rb + wb_ - 1) / wb_;
            for (rr = 0; rr < rows; rr = rr + 1)
                for (bb = 0; bb < rb; bb = bb + 1) begin
                    wd  = mode ? w + (bb / wb_) * rows + rr : w + rr * wpr + bb / wb_;
                    byt = bb % wb_;
                    if (m == M_A) sh_a[wd][8*byt +: 8] = mem[d - MEM_BASE + rr*pitch + bb];
                    if (m == M_B) sh_b[wd][8*byt +: 8] = mem[d - MEM_BASE + rr*pitch + bb];
                    if (m == M_C) sh_c[wd][8*byt +: 8] = mem[d - MEM_BASE + rr*pitch + bb];
                end
        end
    endtask
    task ref_st(input [31:0] d, input [3:0] m, input integer w, input integer rows, input integer rb,
                input integer pitch);
        integer rr, bb, wd, byt, wb_, wpr;
        begin
            wb_ = m == M_C ? 4*D : D;
            wpr = (rb + wb_ - 1) / wb_;
            for (rr = 0; rr < rows; rr = rr + 1)
                for (bb = 0; bb < rb; bb = bb + 1) begin
                    wd  = w + rr * wpr + bb / wb_;
                    byt = bb % wb_;
                    rd_sh[d - RD + rr*pitch + bb] = m == M_A ? sh_a[wd][8*byt +: 8] :
                                                    m == M_B ? sh_b[wd][8*byt +: 8] : sh_c[wd][8*byt +: 8];
                end
        end
    endtask
    task ref_ex(input integer a, input integer b, input integer c, input integer kt, input integer acc,
                input integer R, input integer bstep, input integer cstep, input integer crow);
        reg signed [31:0] res [0:D-1][0:D-1];
        integer i, j, k, r, bb, cc;
        begin
            for (r = 0; r < R; r = r + 1) begin                 // tiles run in order
                bb = b + r * bstep;
                cc = c + r * cstep;
                for (i = 0; i < D; i = i + 1) for (j = 0; j < D; j = j + 1) begin
                    res[i][j] = acc ? $signed(sh_c[cc + i*crow][32*j +: 32]) : 0;
                    for (k = 0; k < kt*D; k = k + 1)
                        res[i][j] = res[i][j] + $signed(sh_a[a + (k/D)*D + i][8*(k%D) +: 8]) *
                                                $signed(sh_b[bb + k][8*j +: 8]);
                end
                for (i = 0; i < D; i = i + 1) for (j = 0; j < D; j = j + 1) sh_c[cc + i*crow][32*j +: 32] = res[i][j];
            end
        end
    endtask

    // words near the start, the bank boundary and the end: frequent bank conflicts
    function integer pick(input integer depth, input integer span);
        integer v;
        begin
            case (rnd(2))
                0: v = rnd(48);
                1: v = depth/2 - 24 + rnd(48);
                default: v = depth - span - rnd(24);
            endcase
            if (v > depth - span) v = depth - span;
            if (v < 0) v = 0;
            pick = v;
        end
    endfunction

    integer n_ld, n_ex, n_st, n_ve;
    task random_stream(input integer ncmd);
        integer c, typ, m, mode, rows, rb, pitch, w, span, d, kt, a, b, cc, acc, er, ebs, ecs, ecr;
        integer vit, vot, vgr, vop, vm1, vm2, vmd, vw1, vw2, vwd, vper;
        begin
            n_ld = 0; n_ex = 0; n_st = 0; n_ve = 0;
            for (ri = 0; ri < SW; ri = ri + 1) begin
                for (rj = 0; rj < D/4; rj = rj + 1) begin
                    sh_a[ri][32*rj +: 32] = $random(seed);
                    sh_b[ri][32*rj +: 32] = $random(seed);
                end
                if (ri < SW/2) begin dut.spad_a.bank[0].ram.ram_block[ri] = sh_a[ri]; dut.spad_b.bank[0].ram.ram_block[ri] = sh_b[ri]; end
                else begin dut.spad_a.bank[1].ram.ram_block[ri - SW/2] = sh_a[ri]; dut.spad_b.bank[1].ram.ram_block[ri - SW/2] = sh_b[ri]; end
            end
            for (ri = 0; ri < CW; ri = ri + 1) begin
                for (rj = 0; rj < D; rj = rj + 1) sh_c[ri][32*rj +: 32] = $random(seed) >>> 6;
                if (ri < CW/2) dut.accm.bank[0].ram.ram_block[ri] = sh_c[ri]; else dut.accm.bank[1].ram.ram_block[ri - CW/2] = sh_c[ri];
            end
            for (ri = 0; ri < 32768; ri = ri + 1) begin
                mem[RS - MEM_BASE + ri] = $random(seed);
                mem[RD - MEM_BASE + ri] = $random(seed);
                rd_sh[ri] = mem[RD - MEM_BASE + ri];
            end
            for (c = 0; c < ncmd; c = c + 1) begin
                typ = rnd(12);
                if (typ <= 3) begin                                   // LD
                    m = 1 + rnd(2);
                    mode = m != M_C && rnd(2) == 0;
                    rows = 1 + rnd(7); rb = 8 * (1 + rnd(7)); pitch = rb + 8 * rnd(3);
                    if (mode && rb % D) begin                      // INTERLEAVE rows: whole D-byte chunks
                        pitch = pitch + D - rb % D; rb = rb + D - rb % D;
                    end
                    span = mode ? ((rb + D - 1) / D) * rows : rows * ((rb + (m == M_C ? 4*D : D) - 1) / (m == M_C ? 4*D : D));
                    w = pick(m == M_C ? CW : SW, span);
                    d = RS + 8 * rnd((32768 - rows * pitch) / 8 - 1);
                    cfg_ld(rows, rb, pitch, mode);
                    ld(d, m, w);
                    ref_ld(d, m, w, rows, rb, pitch, mode);
                    n_ld = n_ld + 1;
                end else if (typ <= 6) begin                          // EX (M2: 1-4 tiles, random strides)
                    kt = 1 + rnd(3);
                    er = 1 + rnd(3); ebs = rnd(40); ecs = 1 + rnd(9); ecr = 1 + rnd(9);
                    a = pick(SW, kt*D); b = pick(SW, (er - 1)*ebs + kt*D);
                    cc = pick(CW, (er - 1)*ecs + (D-1)*ecr + 1); acc = rnd(1);
                    cfg_ex(er, ebs, ecs, ecr);
                    ex(a, b, cc, kt, acc);
                    ref_ex(a, b, cc, kt, acc, er, ebs, ecs, ecr);
                    n_ex = n_ex + 1;
                end else if (typ >= 10) begin                         // VE (M3)
                    vit = rnd(2); vot = rnd(2); vgr = 1 + rnd(12); vop = rnd(5);
                    if (vop == VMUL && vit == T32) vop = VADD;
                    vm1 = vit == T32 ? M_C : 1 + rnd(1);
                    vm2 = vit == T32 ? M_C : 1 + rnd(1);
                    vmd = vot == T32 ? M_C : 1 + rnd(1);
                    vw1 = pick(vm1 == M_C ? CW : SW, vgr * vwpg(vit));
                    vw2 = pick(vm2 == M_C ? CW : SW, vgr * vwpg(vit));
                    // destination away from the sources (partial overlap is undefined)
                    vwd = (vmd == M_C ? CW : SW) / 4 + rnd(200);
                    vper = rnd(3);
                    vrun(vm1, vw1, vm2, vw2, vmd, vwd, vgr, vop, rnd(1), rnd(1), vit, vot, vper,
                         $signed(rnd(4000)) - 2000, rnd(20), $signed(rnd(200)) - 100,
                         -32'sd2147483648, 32'sd2147483647);
                    ref_ve(vm1, vw1, vm2, vw2, vmd, vwd, vgr, vop, dut.pcpi.v_op[4], dut.pcpi.v_op[5], vit, vot,
                           vper, $signed(dut.pcpi.v_scale), dut.pcpi.v_shift, $signed(dut.pcpi.v_zp),
                           -32'sd2147483648, 32'sd2147483647);
                    n_ve = n_ve + 1;
                end else begin                                        // ST
                    m = 1 + rnd(2);
                    rows = 1 + rnd(3); rb = 8 * (1 + rnd(7)); pitch = rb + 8 * rnd(2);
                    span = rows * ((rb + (m == M_C ? 4*D : D) - 1) / (m == M_C ? 4*D : D));
                    w = pick(m == M_C ? CW : SW, span);
                    d = RD + 8 * rnd((32768 - rows * pitch) / 8 - 1);
                    cfg_st(rows, rb, pitch);
                    st(d, m, w);
                    ref_st(d, m, w, rows, rb, pitch);
                    n_st = n_st + 1;
                end
            end
            fence(0);
            cfg_ex(1, 0, 0, 1);
            if (xst[1]) fail("random stream: sticky error");
            for (ri = 0; ri < SW; ri = ri + 1) begin
                // function results go through regs first: xsim can report a
                // false mismatch when `!==` compares a function result directly
                cmp_h = hw_a(ri); cmp_r = sh_a[ri];
                if (cmp_h !== cmp_r) begin errors = errors + 1; if (errors <= 20) $display("TB ERROR random: SPAD_A[%0d] = %h, expected %h", ri, cmp_h, cmp_r); end
                cmp_h = hw_b(ri); cmp_r = sh_b[ri];
                if (cmp_h !== cmp_r) begin errors = errors + 1; if (errors <= 20) $display("TB ERROR random: SPAD_B[%0d] = %h, expected %h", ri, cmp_h, cmp_r); end
            end
            for (ri = 0; ri < CW; ri = ri + 1) begin
                cmp_h = hw_c(ri); cmp_r = sh_c[ri];
                if (cmp_h !== cmp_r) begin errors = errors + 1; if (errors <= 20) $display("TB ERROR random: ACC[%0d] = %h, expected %h", ri, cmp_h, cmp_r); end
            end
            for (ri = 0; ri < 32768; ri = ri + 1)
                if (mem[RD - MEM_BASE + ri] !== rd_sh[ri]) begin
                    errors = errors + 1; if (errors <= 20) $display("TB ERROR random: DDR RD+%0d", ri);
                end
        end
    endtask


    // ============================================== vector engine (M3)
    task v_op(input [2:0] f3, input [31:0] rs1, input [31:0] rs2);
        begin
            pcpi_exec(custom0(f3, 7'd2), rs1, rs2);
            if (pc_cycles < 0) fail("vector instruction timed out (would trap)");
        end
    endtask
    task vcfg(input [7:0] key, input [31:0] v); v_op(3'd0, key, v); endtask
    // full vector command: all parameters, then vec_run
    task vrun(input [3:0] m1, input integer w1, input [3:0] m2, input integer w2, input [3:0] md, input integer wd,
              input integer groups, input [2:0] op, input relu, input requant, input [1:0] it, input [1:0] ot,
              input integer period, input integer scale, input integer shift, input integer zp,
              input integer lo, input integer hi);
        begin
            vcfg(V_OP, {requant, relu, 1'b0, op}); vcfg(V_LEN, groups * D); vcfg(V_DST, {md, 12'd0, wd[15:0]});
            vcfg(V_TYPES, {ot, it}); vcfg(V_MOD, period); vcfg(V_SCALE, scale); vcfg(V_SHIFT, shift);
            vcfg(V_ZP, zp); vcfg(V_LO, lo); vcfg(V_HI, hi);
            v_op(3'd1, {m1, 12'd0, w1[15:0]}, {m2, 12'd0, w2[15:0]});
        end
    endtask

    function integer vwpg(input [1:0] t); vwpg = t == T16 ? 2 : 1; endfunction
    function [AW-1:0] shg(input [3:0] m, input integer w);
        shg = m == M_A ? {{AW-8*D{1'b0}}, sh_a[w]} : m == M_B ? {{AW-8*D{1'b0}}, sh_b[w]} : sh_c[w];
    endfunction
    function signed [31:0] relem(input [3:0] m, input integer base, input [1:0] t, input integer e);
        reg [AW-1:0] w0, w1;
        begin
            w0 = shg(m, base); w1 = shg(m, base + 1);
            if (t == T8)       relem = $signed(w0[8*e +: 8]);
            else if (t == T16) relem = e < D/2 ? $signed(w0[16*e +: 16]) : $signed(w1[16*(e - D/2) +: 16]);
            else               relem = $signed(w0[32*e +: 32]);
        end
    endfunction
    function signed [63:0] clampv(input signed [63:0] x, input signed [63:0] lo_, input signed [63:0] hi_);
        clampv = x < lo_ ? lo_ : x > hi_ ? hi_ : x;
    endfunction
    // the arithmetic of one element (also the golden model of the quantized GEMM)
    function signed [31:0] vcalc(input signed [31:0] a, input signed [31:0] b, input [2:0] op, input relu,
                                 input requant, input [1:0] ot, input integer scale, input integer shift,
                                 input integer zp, input integer lo, input integer hi);
        reg signed [63:0] r, q;
        begin
            case (op)
                VADD: r = clampv(a + b, -64'sd2147483648, 64'sd2147483647);
                VSUB: r = clampv(a - b, -64'sd2147483648, 64'sd2147483647);
                VMUL: r = $signed(a[15:0]) * $signed(b[15:0]);
                VMAX: r = a > b ? a : b;
                VMIN: r = a < b ? a : b;
                default: r = a;
            endcase
            if (relu && r < 0) r = 0;
            q = requant ? ((r * $signed(scale[15:0]) + (shift == 0 ? 0 : (64'sd1 <<< (shift - 1)))) >>> shift) + zp : r;
            q = clampv(q, lo, hi);
            vcalc = clampv(q, ot == T8 ? -128 : ot == T16 ? -32768 : -64'sd2147483648,
                              ot == T8 ?  127 : ot == T16 ?  32767 :  64'sd2147483647);
        end
    endfunction
    reg signed [31:0] vres [0:4095][0:D-1];
    task ref_ve(input [3:0] m1, input integer w1, input [3:0] m2, input integer w2, input [3:0] md, input integer wd,
                input integer groups, input [2:0] op, input relu, input requant, input [1:0] it, input [1:0] ot,
                input integer period, input integer scale, input integer shift, input integer zp,
                input integer lo, input integer hi);
        integer g, g2, e;
        reg signed [31:0] ea, eb;
        reg [AW-1:0] x0, x1;
        begin
            for (g = 0; g < groups; g = g + 1) begin
                g2 = period == 0 ? g : period == 1 ? 0 : g % period;
                for (e = 0; e < D; e = e + 1) begin
                    // one relem call per statement: xsim returns the last call's
                    // value for every call of a static function in one expression
                    ea = relem(m1, w1 + g * vwpg(it), it, e);
                    eb = op == VCOPY ? 0 : relem(m2, w2 + g2 * vwpg(it), it, e);
                    vres[g][e] = vcalc(ea, eb, op, relu, requant, ot, scale, shift, zp, lo, hi);
                end
            end
            for (g = 0; g < groups; g = g + 1) begin
                x0 = 0; x1 = 0;
                for (e = 0; e < D; e = e + 1)
                    if (ot == T8)       x0[8*e +: 8] = vres[g][e][7:0];
                    else if (ot == T16) begin if (e < D/2) x0[16*e +: 16] = vres[g][e][15:0]; else x1[16*(e - D/2) +: 16] = vres[g][e][15:0]; end
                    else                x0[32*e +: 32] = vres[g][e];
                if (md == M_A) begin sh_a[wd + g * vwpg(ot)] = x0[8*D-1:0]; if (ot == T16) sh_a[wd + 2*g + 1] = x1[8*D-1:0]; end
                if (md == M_B) begin sh_b[wd + g * vwpg(ot)] = x0[8*D-1:0]; if (ot == T16) sh_b[wd + 2*g + 1] = x1[8*D-1:0]; end
                if (md == M_C) sh_c[wd + g] = x0;
            end
        end
    endtask

    // ---- quantized GEMM: int8 C = requant(relu(A x B + bias)), fused on the chip
    //      mode 0: VE adds the bias vector (period N/8) + RELU + REQUANT
    //      mode 1: bias rows preloaded into ACC, EX accumulates, VE does RELU + REQUANT (COPY)
    integer qgemm_cycles, qgemm_cycles_first;
    task qgemm(input integer M, input integer N, input integer K, input integer mode,
               input integer scale, input integer shift, input integer zp);
        localparam [31:0] QA = MEM_BASE + 32'h20000, QB = MEM_BASE + 32'h22000,
                          QC = MEM_BASE + 32'h24000, QBIAS = MEM_BASE + 32'h28000;
        integer t0, i, j, k, bk, vb;
        reg signed [31:0] acc32;
        reg signed [31:0] qexp;
        begin
            for (i = 0; i < M; i = i + 1) for (k = 0; k < K; k = k + 1) begin
                GA[i][k] = $random(seed); mem[QA - MEM_BASE + i*K + k] = GA[i][k];
            end
            for (k = 0; k < K; k = k + 1) for (j = 0; j < N; j = j + 1) begin
                GB[k][j] = $random(seed); mem[QB - MEM_BASE + k*N + j] = GB[k][j];
            end
            for (j = 0; j < N; j = j + 1) begin                  // bias vector (and its M replicated rows)
                GBIAS[0][j] = $random(seed) >>> 14;
                for (i = 0; i < M; i = i + 1) begin
                    GBIAS[i][j] = GBIAS[0][j];
                    {mem[QBIAS - MEM_BASE + 4*(i*N + j) + 3], mem[QBIAS - MEM_BASE + 4*(i*N + j) + 2],
                     mem[QBIAS - MEM_BASE + 4*(i*N + j) + 1], mem[QBIAS - MEM_BASE + 4*(i*N + j)]} = GBIAS[0][j];
                end
            end
            poison(QC, M*N + 16);
            t0 = $time;
            vb = CW/2 - N/D;                                     // bias vector: end of ACC bank 0
            cfg_ld(K, N, N, 1); ld(QB, M_B, 0);                  // resident B
            if (mode == 0) begin cfg_ld(1, 4*N, 4*N, 0); ld(QBIAS, M_C, vb); end
            cfg_ex(N/D, K, 1, N/D);
            cfg_st(D, N, N);                                     // int8 strip: D rows of N bytes
            for (i = 0; i < M/D; i = i + 1) begin
                bk = i % 2;
                cfg_ld(D, K, K, 1);
                ld(QA + i*D*K, M_A, bk * SW/2);
                if (mode == 1) begin cfg_ld(D, 4*N, 4*N, 0); ld(QBIAS + i*D*4*N, M_C, bk * CW/2); end
                ex(bk * SW/2, 0, bk * CW/2, K/D, mode == 1);
                if (mode == 0)
                    vrun(M_C, bk * CW/2, M_C, vb, M_A, bk * SW/2 + SW/4, N, VADD, 1, 1, T32, T8,
                         N/D, scale, shift, zp, -32'sd2147483648, 32'sd2147483647);
                else
                    vrun(M_C, bk * CW/2, M_C, 0, M_A, bk * SW/2 + SW/4, N, VCOPY, 1, 1, T32, T8,
                         0, scale, shift, zp, -32'sd2147483648, 32'sd2147483647);
                st(QC + i*D*N, M_A, bk * SW/2 + SW/4);
            end
            fence(0);
            qgemm_cycles = ($time - t0) / 10;
            cfg_ex(1, 0, 0, 1);
            if (xst[1]) fail("quantized GEMM: sticky error");
            for (i = 0; i < M; i = i + 1) for (j = 0; j < N; j = j + 1) begin
                acc32 = GBIAS[i][j];
                for (k = 0; k < K; k = k + 1) acc32 = acc32 + GA[i][k] * GB[k][j];
                qexp = vcalc(acc32, 0, VCOPY, 1, 1, T8, scale, shift, zp, -32'sd2147483648, 32'sd2147483647);
                if ($signed(mem[QC - MEM_BASE + i*N + j]) !== qexp[7:0] && mem[QC - MEM_BASE + i*N + j] !== qexp[7:0]) begin
                    errors = errors + 1;
                    if (errors <= 20) $display("TB ERROR qGEMM mode %0d C[%0d][%0d] = %0d, expected %0d",
                                               mode, i, j, $signed(mem[QC - MEM_BASE + i*N + j]), qexp);
                end
            end
            for (i = M*N; i < M*N + 16; i = i + 1)
                if (mem[QC - MEM_BASE + i] !== 8'hA5) fail("quantized GEMM: guard overwritten");
        end
    endtask

    // ============================================ performance counters (P1)
    // docs/perf_counters_and_desc_dma_plan.md §1.6: exact invariants after
    // GEMMs with known command / tile / byte counts
    localparam integer NPC = 32;
    localparam integer PC_CYCLES = 0, PC_CMD_LD = 1, PC_CMD_ST = 2, PC_CMD_EX = 3, PC_CMD_VE = 4,
                       PC_HAZ_LD = 7, PC_HAZ_ST = 8, PC_HAZ_EX = 9, PC_HAZ_VE = 10,
                       PC_STARVE = 12, PC_ALL_IDLE = 13,
                       PC_EX_STEP = 14, PC_EX_USEFUL = 15, PC_EX_TILES = 17,
                       PC_LD_BUSY = 18, PC_LD_BEATS = 19, PC_ST_BUSY = 21, PC_ST_BEATS = 22,
                       PC_VE_ACTIVE = 24, PC_VE_GROUPS = 27;
    reg  [31:0] pc_a [0:NPC-1];
    reg  [31:0] pc_b [0:NPC-1];
    reg  [31:0] pc_csr;
    integer     perf_t0, perf_win, pi, perf_checks = 0;

    task perf_ctl(input [1:0] f);                    // mat_perf control: [0] clear, [1] enable
        begin
            pcpi_exec(custom0(3'd5, 7'd1), 32'h8000_0000, {30'd0, f});
            if (pc_cycles < 0) fail("mat_perf control timed out (would trap)");
            if (pc_rd !== (PERF != 0 ? NPC : 0)) fail("mat_perf control: wrong counter count");
        end
    endtask
    task perf_snap;                                   // all counters through mat_perf
        for (pi = 0; pi < NPC; pi = pi + 1) begin
            pcpi_exec(custom0(3'd5, 7'd1), pi, 0);
            if (pc_cycles < 0) fail("mat_perf read timed out (would trap)");
            pc_a[pi] = pc_rd;
        end
    endtask
    task perf_expect(input integer idx, input [31:0] want, input [8*24-1:0] what);
        begin
            perf_checks = perf_checks + 1;
            if (pc_a[idx] !== want) begin
                errors = errors + 1;
                if (errors <= 20) $display("TB ERROR perf %0s = %0d, expected %0d", what, pc_a[idx], want);
            end
        end
    endtask
    task perf_true(input cond, input [8*48-1:0] what);
        begin
            perf_checks = perf_checks + 1;
            if (!cond) fail(what);
        end
    endtask
    // checks common to every measured window: counters frozen, mirror == PCPI,
    // reserved slots zero, sane relations, CYCLES == the window seen by the testbench
    task perf_common;
        begin
            for (pi = 0; pi < NPC; pi = pi + 1) pc_b[pi] = pc_a[pi];
            repeat (50) @(posedge aclk);
            perf_snap;                                            // frozen: unchanged
            for (pi = 0; pi < NPC; pi = pi + 1)
                perf_true(pc_a[pi] === pc_b[pi], "perf: counter changed while frozen");
            for (pi = 0; pi < NPC; pi = pi + 1) begin              // CSR mirror
                csr_read(8'h40 + 4 * pi, pc_csr);
                perf_true(pc_csr === pc_a[pi], "perf: CSR mirror differs from mat_perf");
            end
            if (PERF == 0) begin
                for (pi = 0; pi < NPC; pi = pi + 1) perf_true(pc_a[pi] === 0, "perf: PERF=0 counter not zero");
            end else begin
                for (pi = 28; pi < NPC; pi = pi + 1) perf_true(pc_a[pi] === 0, "perf: reserved counter not zero");
                perf_true(pc_a[PC_CYCLES] + 12 >= perf_win && pc_a[PC_CYCLES] <= perf_win + 12,
                          "perf: CYCLES differs from the measured window");
                perf_true(pc_a[PC_LD_BUSY] >= pc_a[PC_LD_BEATS], "perf: LD_BUSY < LD_BEATS");
                perf_true(pc_a[PC_ST_BUSY] >= pc_a[PC_ST_BEATS], "perf: ST_BUSY < ST_BEATS");
                perf_true(pc_a[PC_CYCLES] >= pc_a[PC_EX_STEP], "perf: CYCLES < EX_STEP");
                perf_true(pc_a[PC_HAZ_LD] + pc_a[PC_HAZ_ST] + pc_a[PC_HAZ_EX] + pc_a[PC_HAZ_VE] +
                          pc_a[PC_STARVE] + pc_a[PC_ALL_IDLE] <= pc_a[PC_CYCLES],
                          "perf: head states exceed CYCLES");
            end
        end
    endtask

    // int32 GEMM: LD B + per strip {LD A, [LD bias], EX (N/D tiles), ST}
    task perf_gemm(input integer M, input integer N, input integer K, input integer use_bias);
        integer strips, tiles;
        begin
            strips = M / D; tiles = strips * (N / D);
            perf_ctl(2'b11);                                      // clear + enable
            perf_t0 = $time;
            gemm(M, N, K, MEM_BASE + 32'h20000, MEM_BASE + 32'h22000, MEM_BASE + 32'h24000,
                 use_bias, MEM_BASE + 32'h28000);
            perf_win = ($time - perf_t0) / 10;
            perf_ctl(2'b00);                                      // freeze
            perf_snap;
            if (PERF != 0) begin
                perf_expect(PC_EX_TILES,  tiles,                       "EX_TILES");
                perf_expect(PC_EX_USEFUL, tiles * K,                   "EX_USEFUL");
                perf_expect(PC_EX_STEP,   tiles * (K + 2 * (D - 1)),   "EX_STEP");
                perf_expect(PC_LD_BEATS,  (K * N + M * K + (use_bias ? 4 * M * N : 0)) / 8, "LD_BEATS");
                perf_expect(PC_ST_BEATS,  4 * M * N / 8,               "ST_BEATS");
                perf_expect(PC_CMD_LD,    1 + strips * (use_bias ? 2 : 1), "CMD_LD");
                perf_expect(PC_CMD_EX,    strips,                      "CMD_EX");
                perf_expect(PC_CMD_ST,    strips,                      "CMD_ST");
                perf_expect(PC_CMD_VE,    0,                           "CMD_VE");
                perf_expect(PC_VE_ACTIVE, 0,                           "VE_ACTIVE");
                perf_true(pc_a[PC_EX_USEFUL] * D * D == M * N * K, "perf: EX_USEFUL x D^2 != M*N*K");
            end
            perf_common;
        end
    endtask

    // int8 GEMM, mode 0: LD B, LD bias vector, per strip {LD A, EX, VE (N groups), ST (int8)}
    task perf_qgemm(input integer M, input integer N, input integer K);
        integer strips, tiles;
        begin
            strips = M / D; tiles = strips * (N / D);
            perf_ctl(2'b11);
            perf_t0 = $time;
            qgemm(M, N, K, 0, 181, 15, -3);
            perf_win = ($time - perf_t0) / 10;
            perf_ctl(2'b00);
            perf_snap;
            if (PERF != 0) begin
                perf_expect(PC_EX_TILES,  tiles,                       "EX_TILES");
                perf_expect(PC_EX_USEFUL, tiles * K,                   "EX_USEFUL");
                perf_expect(PC_CMD_VE,    strips,                      "CMD_VE");
                perf_expect(PC_VE_GROUPS, strips * N,                  "VE_GROUPS");
                perf_expect(PC_CMD_LD,    2 + strips,                  "CMD_LD");
                perf_expect(PC_LD_BEATS,  (K * N + 4 * N + M * K) / 8, "LD_BEATS");
                perf_expect(PC_ST_BEATS,  M * N / 8,                   "ST_BEATS");
                perf_true(pc_a[PC_VE_ACTIVE] >= pc_a[PC_VE_GROUPS], "perf: VE_ACTIVE < VE_GROUPS");
            end
            perf_common;
        end
    endtask

    // directed hazard: an EX that reads the bank a preceding LD writes waits
    // at the head (RAW) - only HAZ_EX may count; one LD and one EX dispatched
    task perf_hazard;
        begin
            fence(0);
            cfg_ex(1, 0, 0, 1);
            perf_ctl(2'b11);
            perf_t0 = $time;
            cfg_ld(D, 64 * D, 64 * D, 1);                          // A strip, K = 64 D
            ld(RS, M_A, 0);
            ex(0, 0, 0, 64, 0);                                    // reads SPAD_A bank 0
            fence(0);
            perf_win = ($time - perf_t0) / 10;
            perf_ctl(2'b00);
            perf_snap;
            if (PERF != 0) begin
                perf_true(pc_a[PC_HAZ_EX] > 0, "perf: HAZ_EX not counted for a RAW on SPAD_A");
                perf_expect(PC_HAZ_LD, 0, "HAZ_LD");
                perf_expect(PC_HAZ_ST, 0, "HAZ_ST");
                perf_expect(PC_HAZ_VE, 0, "HAZ_VE");
                perf_expect(PC_CMD_LD, 1, "CMD_LD");
                perf_expect(PC_CMD_EX, 1, "CMD_EX");
                perf_expect(PC_EX_TILES, 1, "EX_TILES");
                perf_expect(PC_LD_BEATS, 64 * D * D / 8, "LD_BEATS");
            end
            perf_common;
        end
    endtask

    // control paths: mat_perf clear, CSR PERF_CTRL enable / clear / readback
    task perf_ctl_tests;
        begin
            perf_ctl(2'b01);                                      // clear only
            perf_snap;
            for (pi = 0; pi < NPC; pi = pi + 1) perf_true(pc_a[pi] === 0, "perf: clear left a counter non-zero");
            csr_write(8'h3C, 32'd2);                              // CSR: enable
            csr_read(8'h3C, pc_csr);
            perf_true(pc_csr === (PERF != 0 ? 32'd2 : 32'd0), "perf: PERF_CTRL readback (enabled)");
            repeat (100) @(posedge aclk);
            csr_write(8'h3C, 32'd0);                              // CSR: freeze
            csr_read(8'h40, pc_csr);
            if (PERF != 0) perf_true(pc_csr >= 100 && pc_csr < 200, "perf: CSR-enabled CYCLES out of range");
            else           perf_true(pc_csr === 0, "perf: PERF=0 CYCLES not zero");
            csr_write(8'h3C, 32'd1);                              // CSR: clear
            csr_read(8'h40, pc_csr);
            perf_true(pc_csr === 0, "perf: CSR clear did not clear CYCLES");
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
        if (status !== {20'd0, 4'd3, 5'd0, 3'b101}) begin fail("read SLVERR: wrong STATUS"); $display("   STATUS = %08x, timeout %0d", status, timeout); end
        if (mem32(c_addr) !== 32'hA5A5A5A5) fail("read SLVERR: C was written anyway");

        // --- error: SLVERR on write response
        inject_bresp = 1;
        run(a_addr, b_addr, c_addr, 3'b000);
        wait_done(0);
        inject_bresp = 0;
        if (status !== {20'd0, 4'd4, 5'd0, 3'b101}) begin fail("write SLVERR: wrong STATUS"); $display("   STATUS = %08x, timeout %0d", status, timeout); end

        // --- recovery after errors
        poison(c_addr, 272);
        run(a_addr, b_addr, c_addr, 3'b000);
        wait_done(0);
        if (status !== 32'h1) begin fail("recovery run: wrong STATUS"); $display("   STATUS = %08x, timeout %0d", status, timeout); end
        check_c(cs, c_addr);

        // ================= custom-instruction (PCPI) path =================
        // --- all golden vectors via mat_trigger / mat_wait, no CSR access
        for (cs = 0; cs < NC; cs = cs + 1) begin
            // own 1 KB block per case: A +0x000, B +0x080, C +0x100 (+ small offsets)
            a_addr = MEM_BASE + 32'h10000 + cs * 32'h400 + (cs % 8) * 8;
            b_addr = MEM_BASE + 32'h10000 + cs * 32'h400 + 32'h80 + (cs % 4) * 8;
            c_addr = MEM_BASE + 32'h10000 + cs * 32'h400 + 32'h100 + (cs % 3) * 8;
            load_matrix(a_addr, 8 * cs, 0);
            load_matrix(b_addr, 8 * cs, 1);
            poison(c_addr, 272);
            put_desc(MEM_BASE + 32'h18000 + 16 * cs, a_addr, b_addr, DIM_888);
            mat_op(F_TRIGGER, MEM_BASE + 32'h18000 + 16 * cs, c_addr);
            if (pc_wr) fail("mat_trigger wrote rd");
            mat_op(F_WAIT, 0, 0);
            if (!pc_wr || pc_rd !== 32'h1) fail("mat_wait: status not done/ok");
            check_c(cs, c_addr);
            mat_op(F_CYCLES, 0, 0);
            if (!pc_wr || pc_rd == 0) fail("mat_cycles returned 0");
        end
        expect_csr(SRC_A, a_addr, "SRC_A after mat_trigger (from descriptor)");
        expect_csr(DST, c_addr, "DST after mat_trigger");

        // --- mat_status while busy, then trigger-while-busy stalls until idle
        cs = 3;
        a_addr = MEM_BASE + 32'hC000; b_addr = MEM_BASE + 32'hC100; c_addr = MEM_BASE + 32'hC200;
        load_matrix(a_addr, 8 * cs, 0);
        load_matrix(b_addr, 8 * cs, 1);
        poison(c_addr, 272);
        poison(MEM_BASE + 32'hC400, 272);
        put_desc(MEM_BASE + 32'hF000, a_addr, b_addr, DIM_888);
        mat_op(F_TRIGGER, MEM_BASE + 32'hF000, c_addr);
        mat_op(F_STATUS, 0, 0);
        if (pc_rd[1] !== 1'b1 || pc_cycles > 2) fail("mat_status: not immediate / busy not set");
        mat_op(F_TRIGGER, MEM_BASE + 32'hF000, MEM_BASE + 32'hC400);   // same A/B, second C
        if (pc_cycles < 40) fail("mat_trigger while busy did not stall");
        mat_op(F_WAIT, 0, 0);
        if (pc_rd !== 32'h1) fail("second job status");
        check_c(cs, c_addr);
        check_c(cs, MEM_BASE + 32'hC400);

        // --- errors: misaligned descriptor, bad DIM in descriptor
        mat_op(F_TRIGGER, MEM_BASE + 32'hF008, c_addr);
        mat_op(F_WAIT, 0, 0);
        if (pc_rd !== {20'd0, 4'd2, 5'd0, 3'b101}) fail("misaligned descriptor: wrong status");
        put_desc(MEM_BASE + 32'hF010, a_addr, b_addr, {2'b0, 10'd16, 10'd8, 10'd8});
        mat_op(F_TRIGGER, MEM_BASE + 32'hF010, c_addr);
        mat_op(F_WAIT, 0, 0);
        if (pc_rd !== {20'd0, 4'd1, 5'd0, 3'b101}) fail("descriptor DIM error: wrong status");
        csr_write(DIM, DIM_888);   // the descriptor also loaded DIM into the CSR

        // --- mat_reset clears status
        mat_op(F_RESET, 0, 0);
        mat_op(F_STATUS, 0, 0);
        if (pc_rd !== 32'h0) fail("mat_reset did not clear status");

        // --- unknown encodings are not claimed (core must trap)
        pcpi_exec(custom0(3'd5, 7'd0), 0, 0);
        if (pc_cycles >= 0) fail("funct3=5 was answered");
        pcpi_exec(custom0(3'd0, 7'd3), 0, 0);
        if (pc_cycles >= 0) fail("funct7=3 was answered");
        pcpi_exec(custom0(3'd2, 7'd2), 0, 0);
        if (pc_cycles >= 0) fail("funct7=2 funct3=2 was answered");
        pcpi_exec(32'h02B50533, 0, 0);            // mul a0,a0,a1 (not custom-0)
        if (pc_cycles >= 0) fail("standard MUL claimed by matmul_pcpi");

        // --- CSR start, then mat_trigger while the CSR job runs
        poison(c_addr, 272);
        poison(MEM_BASE + 32'hC400, 272);
        run(a_addr, b_addr, c_addr, 3'b000);
        mat_op(F_TRIGGER, MEM_BASE + 32'hF000, MEM_BASE + 32'hC400);
        mat_op(F_WAIT, 0, 0);
        if (pc_rd !== 32'h1) fail("CSR + PCPI mix: status");
        check_c(cs, c_addr);
        check_c(cs, MEM_BASE + 32'hC400);

        // =============================================== new ISA: GEMM
        // shapes in units of D (D = 8: 24x16x40, 16x32x128, 8x8x8, 32x32x64)
        gemm(3*D, 2*D, 5*D, MEM_BASE + 32'h20000, MEM_BASE + 32'h22000, MEM_BASE + 32'h24000, 0, 0);
        gemm(2*D, 4*D, 128, MEM_BASE + 32'h20000, MEM_BASE + 32'h22000, MEM_BASE + 32'h24000, 1, MEM_BASE + 32'h28000);
        gemm(D, D, D,       MEM_BASE + 32'h20000, MEM_BASE + 32'h22000, MEM_BASE + 32'h24000, 1, MEM_BASE + 32'h28000);
        gemm(4*D, 4*D, 8*D, MEM_BASE + 32'h20000, MEM_BASE + 32'h22000, MEM_BASE + 32'h24000, 0, 0);
        gemm_cycles_main = gemm_cycles;                     // (the perf tests run more GEMMs)

        // ======================================= M3: quantized GEMMs
        // D = 8: 32x32x64, 16x48x40, 24x16x128
        qgemm(4*D, 4*D, 8*D, 0, 181, 15, -3);
        qgemm_cycles_first = qgemm_cycles;
        qgemm(2*D, 6*D, 5*D, 1, -97, 12, 7);
        qgemm(3*D, 2*D, 128, 0, 1, 0, 0);

        // ===================================== scoreboard: random stream
        // ============================= performance counters (P1)
        perf_gemm(4*D, 4*D, 8*D, 0);
        perf_gemm(3*D, 2*D, 5*D, 1);
        perf_qgemm(4*D, 4*D, 8*D);
        perf_hazard;
        perf_ctl_tests;

        random_stream(120);

        // ================================================ new-ISA errors
        cfg_ld(2, 12, 16, 0);                                 // row_bytes not a multiple of 8
        ld(MEM_BASE + 32'h30000, M_A, 100);
        fence(0);
        if (xst[1] !== 1'b1 || xst[11:8] !== 4'd1 || xst[15:12] !== 4'd0) fail("bad LD shape: wrong ext status");
        rk = hw_a(200);
        cfg_ld(1, 8, 8, 0);
        ld(MEM_BASE + 32'h30000, M_A, 200);                   // dropped while the error is set
        fence(0);
        cmp_h = hw_a(200);
        if (cmp_h[8*D-1:0] !== rk) fail("command executed after a sticky error");
        mat_op(F_RESET, 0, 0);
        fence(0);
        if (xst[1]) fail("mat_reset did not clear the sticky error");
        ld(MEM_BASE + 32'h30000, M_A, 200);
        fence(0);
        cmp_h = hw_a(200);                                    // 8 bytes: lane 0 of the word
        if (xst[1] || cmp_h[63:0] !== {mem[32'h30007], mem[32'h30006], mem[32'h30005], mem[32'h30004],
                                       mem[32'h30003], mem[32'h30002], mem[32'h30001], mem[32'h30000]})
            fail("LD after mat_reset");
        ex(SW - 8, 0, 0, 2, 0);                               // A strip past the end of SPAD_A
        fence(0);
        if (xst[11:8] !== 4'd2 || xst[15:12] !== 4'd2) fail("EX range: wrong ext status");
        mat_op(F_RESET, 0, 0);
        ld(MEM_BASE + 32'h30000, 4'd0, 0);                    // no such memory
        fence(0);
        if (xst[11:8] !== 4'd2) fail("bad memory id: wrong ext status");
        mat_op(F_RESET, 0, 0);
        pcpi_exec(custom0(3'd7, 7'd1), 0, 0);                  // 5 is mat_perf (P1), 6 reserved
        if (pc_cycles >= 0) fail("funct7=1 funct3=7 was answered");
        expect_csr(8'h24, (PERF != 0 ? 32'h0010_0000 : 32'd0) + 32'h0001_0000 + D * 256 + D,
                   "CAPS");                                             // [20] perf, 1 port, VL = D, D
        csr_read(8'h28, ext_rd);
        if (ext_rd[0] !== 1'b1 || ext_rd[1] !== 1'b0) fail("EXT_STATUS not idle/ok at the end");

        repeat (20) @(posedge aclk);
        $display("TB %s: %0d errors | legacy: %0d golden cases x (CSR + PCPI), %0d cycles/job avg | D=%0d | new ISA: 4 GEMMs (%0dx%0dx%0d in %0d cycles = %0d MAC/cycle) | random stream %0d LD / %0d EX / %0d ST / %0d VE | 3 quantized GEMMs (%0dx%0dx%0d in %0d cycles) | perf counters: %0d checks | error paths",
                 errors ? "FAIL" : "PASS", errors, NC, total_cycles / NC, D, 4*D, 4*D, 8*D, gemm_cycles_main, 4*D*4*D*8*D / gemm_cycles_main,
                 n_ld, n_ex, n_st, n_ve, 4*D, 4*D, 8*D, qgemm_cycles_first, perf_checks);
        $finish;
    end

    initial begin
        #5_000_000;
        $display("TB FAIL: global timeout");
        $finish;
    end
endmodule
