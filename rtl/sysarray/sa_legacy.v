// Legacy interface of the Phase 2-4 matmul unit on the new datapath
// (docs/double_buffer_design.md §8.5).
//
// - AXI4-Lite CSRs with the Phase 2 map (CTRL, STATUS, SRC_A, SRC_B, DST,
//   DIM, IRQ_STATUS, CYCLES, ID = "MM08") plus CAPS (0x24) and EXT_STATUS
//   (0x28, read-only).
// - A sequencer runs one 8x8x8 job, started by CTRL.start (addresses from
//   the CSRs) or by mat_trigger (16-byte descriptor {A, B, DIM, 0} fetched
//   from DDR into internal registers), as four commands through the normal
//   scheduler: LD A, LD B, EX (Kt = 1), ST C, using the last tile slot of
//   each memory. It fences (waits for all engines) first, so it never races
//   new-style commands. STATUS / error codes / IRQ behave as in Phase 2-4.
//   M1: D = 8 only (D = 16 needs the zero-padded loads of M4).
`timescale 1ns / 1ps
`include "sa_macros.vh"

module sa_legacy #(
    parameter integer D          = 8,
    parameter integer NPORTS     = 1,
    parameter integer VL         = 0,
    parameter integer SPAD_WORDS = 16384,
    parameter integer ACC_WORDS  = 8192
) (
    input  wire                  clk,
    input  wire                  resetn,

    // AXI4-Lite CSR slave
    input  wire [7:0]            s_axi_awaddr,
    input  wire                  s_axi_awvalid,
    output reg                   s_axi_awready,
    input  wire [31:0]           s_axi_wdata,
    input  wire                  s_axi_wvalid,
    output reg                   s_axi_wready,
    output wire [1:0]            s_axi_bresp,
    output reg                   s_axi_bvalid,
    input  wire                  s_axi_bready,
    input  wire [7:0]            s_axi_araddr,
    input  wire                  s_axi_arvalid,
    output reg                   s_axi_arready,
    output reg  [31:0]           s_axi_rdata,
    output wire [1:0]            s_axi_rresp,
    output reg                   s_axi_rvalid,
    input  wire                  s_axi_rready,

    // commands of the sequencer -> scheduler (priority over PCPI)
    output reg                   q_valid,
    input  wire                  q_ready,
    output reg  [`SA_PKT_W-1:0]  q_pkt,
    input  wire                  sched_idle,
    input  wire [31:0]           ext_status,
    output reg                   clear_error,

    // descriptor registers, written by the LD engine (MEM_DESC)
    input  wire                  desc_we,
    input  wire [15:0]           desc_word,
    input  wire [63:0]           desc_data,

    // PCPI funct7 = 0
    input  wire                  leg_trigger,
    input  wire [31:0]           leg_desc,
    input  wire [31:0]           leg_dst,
    output wire                  leg_accept,
    output wire                  leg_busy,
    output wire [31:0]           leg_status,
    output wire [31:0]           leg_cycles,
    input  wire                  leg_reset,
    output wire                  leg_reset_done,

    output wire                  irq
);
    `include "sa_defs.vh"

    localparam [5:0] R_CTRL = 0, R_STATUS = 1, R_SRC_A = 2, R_SRC_B = 3, R_DST = 4, R_DIM = 5,
                     R_IRQ_STATUS = 6, R_CYCLES = 7, R_ID = 8, R_CAPS = 9, R_EXT = 10;
    localparam [31:0] ID_VALUE = 32'h4D4D_3038;                 // "MM08"
    localparam [31:0] DIM_888  = {2'b0, 10'd8, 10'd8, 10'd8};
    localparam [31:0] CAPS     = NPORTS * 65536 + VL * 256 + D;   // [7:0] D, [15:8] VL, [19:16] ports
    localparam [3:0]  ERR_NONE = 0, ERR_DIM = 1, ERR_ADDR = 2, ERR_RRESP = 3, ERR_BRESP = 4;

    // reserved tile slots (last D words of each memory)
    localparam [15:0] SLOT_S = SPAD_WORDS - D;
    localparam [15:0] SLOT_C = ACC_WORDS - D;

    // ---------------------------------------------------------- state
    reg         irq_en, done, error, irq_status;
    reg  [3:0]  err_code;
    reg  [31:0] src_a, src_b, dst, dim, cycles;
    reg  [127:0] desc_reg;

    localparam [3:0] L_IDLE = 0, L_FENCE = 1, L_DESC = 2, L_DESC_W = 3, L_CHECK = 4,
                     L_PUSH = 5, L_RUN = 6, L_ERRW = 7;
    reg  [3:0]  lstate;
    reg  [3:0]  pend_code;        // L_ERRW: error of this job, reported once the engines are idle
    reg         from_desc;
    reg  [31:0] desc_addr;
    reg  [1:0]  pidx;

    wire busy = lstate != L_IDLE;
    wire sched_err = ext_status[1];
    wire [3:0] sched_code = ext_status[11:8];
    wire eng_idle = ext_status[7:4] == 0;

    assign leg_busy   = busy;
    assign leg_status = {20'd0, err_code, 5'd0, error, busy, done};
    assign leg_cycles = cycles;
    assign irq        = irq_status & irq_en;

    // ---------------------------------------------------- CSR slave
    reg        aw_got, w_got;
    reg [7:0]  wr_addr;
    reg [31:0] wr_data;
    wire       csr_we    = aw_got && w_got && !s_axi_bvalid;
    wire [5:0] csr_reg   = wr_addr[7:2];
    wire       csr_start = csr_we && csr_reg == R_CTRL && wr_data[0] && !busy;

    assign s_axi_bresp = 2'b00;
    assign s_axi_rresp = 2'b00;
    assign leg_accept  = leg_trigger && !busy && !csr_start;

    always @(posedge clk) begin
        if (!resetn) begin
            s_axi_awready <= 0; s_axi_wready <= 0; s_axi_bvalid <= 0;
            s_axi_arready <= 0; s_axi_rvalid <= 0; aw_got <= 0; w_got <= 0;
        end else begin
            s_axi_awready <= !aw_got && s_axi_awvalid && !s_axi_awready;
            s_axi_wready  <= !w_got  && s_axi_wvalid  && !s_axi_wready;
            if (s_axi_awvalid && s_axi_awready) begin aw_got <= 1; wr_addr <= s_axi_awaddr; end
            if (s_axi_wvalid && s_axi_wready)   begin w_got  <= 1; wr_data <= s_axi_wdata;  end
            if (csr_we) begin
                aw_got <= 0; w_got <= 0; s_axi_bvalid <= 1;
            end else if (s_axi_bready)
                s_axi_bvalid <= 0;

            s_axi_arready <= s_axi_arvalid && !s_axi_arready && !s_axi_rvalid;
            if (s_axi_arvalid && s_axi_arready) begin
                s_axi_rvalid <= 1;
                case (s_axi_araddr[7:2])
                    R_CTRL:       s_axi_rdata <= {29'd0, irq_en, 2'b00};
                    R_STATUS:     s_axi_rdata <= leg_status;
                    R_SRC_A:      s_axi_rdata <= src_a;
                    R_SRC_B:      s_axi_rdata <= src_b;
                    R_DST:        s_axi_rdata <= dst;
                    R_DIM:        s_axi_rdata <= dim;
                    R_IRQ_STATUS: s_axi_rdata <= {31'd0, irq_status};
                    R_CYCLES:     s_axi_rdata <= cycles;
                    R_ID:         s_axi_rdata <= ID_VALUE;
                    R_CAPS:       s_axi_rdata <= CAPS;
                    R_EXT:        s_axi_rdata <= ext_status;
                    default:      s_axi_rdata <= 32'd0;
                endcase
            end else if (s_axi_rready)
                s_axi_rvalid <= 0;
        end
    end

    // ------------------------------------------------------ sequencer
    function addr_ok(input [31:0] addr, input [12:0] len);
        addr_ok = addr[2:0] == 3'd0 && {1'b0, addr[11:0]} + len <= 13'h1000;
    endfunction

    function [3:0] map_err(input [3:0] xerr);
        map_err = xerr == XERR_RRESP ? ERR_RRESP : xerr == XERR_BRESP ? ERR_BRESP : ERR_ADDR;
    endfunction

    // the four commands of a job
    function [`SA_PKT_W-1:0] job_pkt(input [1:0] idx, input [31:0] a, input [31:0] b, input [31:0] c);
        case (idx)
            2'd0: job_pkt = {1'b1, 2'b00, 32'd8, 16'd8, 16'd8, {MEM_SPAD_A, 12'd0, SLOT_S}, a, CMD_LD};
            2'd1: job_pkt = {1'b1, 2'b00, 32'd8, 16'd8, 16'd8, {MEM_SPAD_B, 12'd0, SLOT_S}, b, CMD_LD};
            2'd2: job_pkt = {70'd0, 1'b0, 12'd1, SLOT_C, SLOT_S, SLOT_S, CMD_EX};
            default: job_pkt = {1'b1, 2'b00, 32'd32, 16'd32, 16'd8, {MEM_ACC, 12'd0, SLOT_C}, c, CMD_ST};
        endcase
    endfunction

    // mat_reset: wait for the sequencer and the engines, then clear everything
    reg reset_done_r;
    assign leg_reset_done = reset_done_r;

    task finish(input [3:0] code);
        begin
            lstate <= L_IDLE; done <= 1; error <= code != ERR_NONE; err_code <= code; irq_status <= 1;
        end
    endtask

    always @(posedge clk) begin
        clear_error  <= 0;
        reset_done_r <= 0;
        if (!resetn) begin
            lstate <= L_IDLE; q_valid <= 0;
            irq_en <= 0; done <= 0; error <= 0; err_code <= 0; irq_status <= 0;
            src_a <= 0; src_b <= 0; dst <= 0; dim <= DIM_888; cycles <= 0;
        end else begin
            if (busy) cycles <= cycles + 1;
            if (desc_we) desc_reg[64*desc_word[0] +: 64] <= desc_data;

            // CSR writes (configuration ignored while busy)
            if (csr_we)
                case (csr_reg)
                    R_CTRL: begin
                        irq_en <= wr_data[2];
                        if (!busy && wr_data[1]) begin
                            done <= 0; error <= 0; err_code <= 0; irq_status <= 0;
                        end
                    end
                    R_SRC_A:      if (!busy) src_a <= wr_data;
                    R_SRC_B:      if (!busy) src_b <= wr_data;
                    R_DST:        if (!busy) dst   <= wr_data;
                    R_DIM:        if (!busy) dim   <= wr_data;
                    R_IRQ_STATUS: if (wr_data[0]) irq_status <= 0;
                    default: ;
                endcase

            if (leg_reset && !busy && eng_idle && !reset_done_r) begin
                done <= 0; error <= 0; err_code <= 0; irq_status <= 0;
                clear_error  <= 1;
                reset_done_r <= 1;
            end

            case (lstate)
                L_IDLE:
                    if (csr_start || leg_accept) begin
                        lstate    <= L_FENCE;
                        from_desc <= !csr_start;
                        desc_addr <= leg_desc;
                        if (!csr_start) dst <= leg_dst;
                        done <= 0; error <= 0; err_code <= 0; cycles <= 0;
                    end
                L_FENCE:
                    if (sched_err)       finish(map_err(sched_code));
                    else if (sched_idle) lstate <= from_desc ? L_DESC : L_CHECK;
                L_DESC:
                    if (desc_addr[3:0] != 0)
                        finish(ERR_ADDR);
                    else if (q_valid && q_ready) begin
                        q_valid <= 0;
                        lstate  <= L_DESC_W;
                    end else begin
                        q_valid <= 1;
                        q_pkt   <= {1'b1, 2'b00, 32'd16, 16'd16, 16'd1, {MEM_DESC, 12'd0, 16'd0},
                                    desc_addr, CMD_LD};
                    end
                L_DESC_W:
                    if (sched_err) begin
                        pend_code <= ERR_RRESP;
                        lstate    <= L_ERRW;
                    end else if (sched_idle) begin
                        src_a  <= desc_reg[31:0];
                        src_b  <= desc_reg[63:32];
                        dim    <= desc_reg[95:64];
                        lstate <= L_CHECK;
                    end
                L_CHECK:
                    if (dim != DIM_888)
                        finish(ERR_DIM);
                    else if (!addr_ok(src_a, 13'd64) || !addr_ok(src_b, 13'd64) || !addr_ok(dst, 13'd256))
                        finish(ERR_ADDR);
                    else begin
                        pidx   <= 0;
                        lstate <= L_PUSH;
                    end
                L_PUSH:
                    if (q_valid && q_ready) begin
                        q_valid <= 0;
                        pidx    <= pidx + 1;
                        if (pidx == 3) lstate <= L_RUN;
                    end else begin
                        q_valid <= 1;
                        q_pkt   <= job_pkt(pidx, src_a, src_b, dst);
                    end
                L_RUN:
                    if (sched_err) begin
                        pend_code <= map_err(sched_code);
                        lstate    <= L_ERRW;
                    end else if (sched_idle)
                        finish(ERR_NONE);
                // a job command failed: the scheduler stops dispatching (the rest of
                // the job stays queued until clear_error flushes it), but another
                // engine of the job may still be running and would set the sticky
                // error again after an early clear - clear once the engines are idle
                L_ERRW:
                    if (eng_idle) begin
                        clear_error <= 1;
                        finish(pend_code);
                    end
                default: lstate <= L_IDLE;
            endcase
        end
    end
endmodule
