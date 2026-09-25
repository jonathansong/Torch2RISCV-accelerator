// PCPI front end: decodes the custom-0 matrix instructions from PicoRV32
// (docs/custom_isa_encoding.md, docs/double_buffer_design.md §8).
//
//   funct7 = 0 (Phase 4, kept): mat_trigger / mat_status / mat_reset /
//              mat_wait / mat_cycles -> legacy block
//   funct7 = 1: mat_cfg (0), mat_load (1), mat_store (2), mat_exec (3),
//              mat_fence (4) -> command queue / scheduler;
//              mat_perf (5) -> performance counters (read / control);
//              mat_submit (6) -> descriptor fetch unit (list, count). While a
//              list runs, queued PCPI commands wait (program order) and
//              mat_fence also waits for the list to finish.
//              mat_notify (7) -> sets NOTIFY_STATUS (host interrupt, L1).
//              mat_cfg keys 11..34 write the fetch unit's BASE0..15 /
//              PARAM0..7 (waiting while a list runs).
//   funct7 = 2: vec_cfg (0), vec_run (1) -> command queue (M3 vector engine)
// Load/store/exec capture the current configuration into the packet, so
// software may reconfigure right after issuing. With a sticky error set,
// new commands are dropped (the core never blocks on a halted queue).
// Unclaimed encodings are not answered: the core raises its illegal-
// instruction trap after the PCPI timeout.
`timescale 1ns / 1ps
`include "sa_macros.vh"

module sa_pcpi #(
    parameter integer D    = 8,            // VL: vec_cfg LEN is in elements
    parameter integer NCNT = 32            // performance counters (0: none), returned by mat_perf control
) (
    input  wire                  clk,
    input  wire                  resetn,

    input  wire                  pcpi_valid,
    input  wire [31:0]           pcpi_insn,
    input  wire [31:0]           pcpi_rs1,
    input  wire [31:0]           pcpi_rs2,
    output reg                   pcpi_wr,
    output reg  [31:0]           pcpi_rd,
    output wire                  pcpi_wait,
    output reg                   pcpi_ready,

    // new commands -> scheduler input (via the arbiter in the top level)
    output reg                   q_valid,
    input  wire                  q_ready,
    output reg  [`SA_PKT_W-1:0]  q_pkt,
    input  wire                  sched_err,
    output reg  [3:0]            fence_mask,
    input  wire                  fence_ok,
    input  wire [31:0]           ext_status,

    // legacy (funct7 = 0)
    output reg                   leg_trigger,     // held until leg_accept
    output reg  [31:0]           leg_desc,
    output reg  [31:0]           leg_dst,
    input  wire                  leg_accept,
    input  wire                  leg_busy,
    input  wire [31:0]           leg_status,
    input  wire [31:0]           leg_cycles,
    output reg                   leg_reset,       // held until leg_reset_done
    input  wire                  leg_reset_done,

    // performance counters (sa_perf): mat_perf
    output reg                   perf_ctl_we,     // one-cycle pulse
    output reg  [1:0]            perf_ctl,        // [0] clear, [1] enable
    output reg  [4:0]            perf_rsel,
    input  wire [31:0]           perf_rdata,
    output wire [1:0]            perf_ev,         // {waiting in mat_fence, stalled on a full queue}

    // descriptor fetch unit (sa_cmdfetch): mat_submit, BASE / PARAM registers
    input  wire                  fetch_busy,
    output reg                   submit,          // one-cycle pulse
    output reg  [31:0]           submit_addr,
    output reg  [31:0]           submit_count,
    output reg                   reg_we,          // one-cycle pulse: mat_cfg keys 11..34
    output reg  [4:0]            reg_idx,         // 0..15 BASE0..15, 16..23 PARAM0..7
    output reg  [31:0]           reg_val,
    // host notification (mat_notify)
    output reg                   notify           // one-cycle pulse
);
    `include "sa_defs.vh"

    wire [6:0] opcode = pcpi_insn[6:0];
    wire [2:0] funct3 = pcpi_insn[14:12];
    wire [6:0] funct7 = pcpi_insn[31:25];
    wire       ours   = opcode == 7'b0001011 &&
                        ((funct7 == 7'd0 && funct3 <= 3'd4) ||
                         funct7 == 7'd1 ||
                         (funct7 == 7'd2 && funct3 <= 3'd1));

    assign pcpi_wait = pcpi_valid && ours;

    // configuration (mat_cfg)
    reg [15:0] ld_rows, ld_rb, st_rows, st_rb;
    reg [31:0] ld_pitch, st_pitch;
    reg [1:0]  ld_mode;
    reg [11:0] ex_rep;                 // tiles per mat_exec (M2)
    reg [15:0] ex_bstep, ex_cstep, ex_crow;
    // vector configuration (vec_cfg), M3
    reg [7:0]  v_op;
    reg [15:0] v_groups;
    reg [31:0] v_dst;
    reg [5:0]  v_types;
    reg [15:0] v_mod, v_scale;
    reg [4:0]  v_shift;
    reg [31:0] v_zp, v_lo, v_hi;
    wire [11:0] ex_rep_m1 = ex_rep == 0 ? 12'd0 : ex_rep - 12'd1;

    localparam [1:0] S_IDLE = 0, S_EXEC = 1, S_DONE = 2;
    reg [1:0]  state;
    reg [1:0]  grp;          // funct7
    reg [2:0]  op;
    reg [31:0] rs1, rs2;

    task respond(input wr, input [31:0] rd);
        begin
            pcpi_ready <= 1; pcpi_wr <= wr; pcpi_rd <= rd; state <= S_DONE;
        end
    endtask

    always @(posedge clk) begin
        pcpi_ready  <= 0;
        pcpi_wr     <= 0;
        perf_ctl_we <= 0;
        submit      <= 0;
        reg_we      <= 0;
        notify      <= 0;
        if (!resetn) begin
            state       <= S_IDLE;
            perf_ctl    <= 0;
            perf_rsel   <= 0;
            q_valid     <= 0;
            leg_trigger <= 0;
            leg_reset   <= 0;
            fence_mask  <= 0;
            ld_rows <= 1; ld_rb <= 8; ld_pitch <= 8; ld_mode <= 0;
            st_rows <= 1; st_rb <= 8; st_pitch <= 8;
            ex_rep <= 1; ex_bstep <= 0; ex_cstep <= 0; ex_crow <= 1;
            v_op <= 0; v_groups <= 1; v_dst <= 0; v_types <= {2'd2, 2'd2}; v_mod <= 0;
            v_scale <= 1; v_shift <= 0; v_zp <= 0; v_lo <= 32'h80000000; v_hi <= 32'h7FFFFFFF;
        end else begin
            case (state)
                S_IDLE:
                    if (pcpi_valid && ours) begin
                        grp   <= funct7[1:0];
                        op    <= funct3;
                        rs1   <= pcpi_rs1;
                        rs2   <= pcpi_rs2;
                        state <= S_EXEC;
                        // build the packet now, from the configuration current at issue
                        q_pkt <= 0;
                        if (funct7[0] && funct3 == 3'd1)
                            q_pkt <= pkt_ld(pcpi_rs1, pcpi_rs2, ld_rows, ld_rb, ld_pitch, ld_mode);
                        if (funct7[0] && funct3 == 3'd2)
                            q_pkt <= pkt_st(pcpi_rs1, pcpi_rs2, st_rows, st_rb, st_pitch);
                        if (funct7[0] && funct3 == 3'd3)       // {acc, Kt, C} {B, A} + EX config
                            q_pkt <= pkt_ex(pcpi_rs1[15:0], pcpi_rs1[31:16], pcpi_rs2[15:0], pcpi_rs2[27:16],
                                            pcpi_rs2[28], ex_rep_m1, ex_bstep, ex_cstep, ex_crow);
                        if (funct7 == 7'd2 && funct3 == 3'd1)  // vec_run: src1, src2 + VE config
                            q_pkt <= pkt_ve(pcpi_rs1, pcpi_rs2, v_dst, v_groups, v_op, v_types, v_mod,
                                            v_scale, v_shift, v_zp, v_lo, v_hi);
                        fence_mask <= pcpi_rs1[3:0];
                        perf_rsel  <= pcpi_rs1[4:0];
                    end
                S_EXEC:
                    if (grp == 2'd2) begin
                        if (op == 3'd0) begin                // vec_cfg
                            case (rs1[7:0])
                                VCFG_OP:       v_op     <= rs2[7:0];
                                VCFG_LEN:      v_groups <= rs2[31:0] >> $clog2(D);   // elements / VL
                                VCFG_DST:      v_dst    <= rs2;
                                VCFG_TYPES:    v_types  <= rs2[5:0];
                                VCFG_SRC2_MOD: v_mod    <= rs2[15:0];
                                VCFG_SCALE:    v_scale  <= rs2[15:0];
                                VCFG_SHIFT:    v_shift  <= rs2[4:0];
                                VCFG_ZP:       v_zp     <= rs2;
                                VCFG_CLAMP_LO: v_lo     <= rs2;
                                VCFG_CLAMP_HI: v_hi     <= rs2;
                                default: ;
                            endcase
                            respond(0, 0);
                        end else if (q_valid && q_ready) begin   // vec_run
                            q_valid <= 0;
                            respond(0, 0);
                        end else if (sched_err) begin
                            q_valid <= 0;
                            respond(0, 0);
                        end else if (!fetch_busy)            // after a running list
                            q_valid <= 1;
                    end else if (grp == 2'd0) begin
                        case (op)
                            3'd0: if (leg_trigger && leg_accept) begin
                                      leg_trigger <= 0;
                                      respond(0, 0);
                                  end else begin
                                      leg_trigger <= 1;
                                      leg_desc    <= rs1;
                                      leg_dst     <= rs2;
                                  end
                            3'd1: respond(1, leg_status);
                            3'd2: if (leg_reset && leg_reset_done) begin
                                      leg_reset <= 0;
                                      respond(0, 0);
                                  end else
                                      leg_reset <= 1;
                            3'd3: if (!leg_busy) respond(1, leg_status);
                            default: respond(1, leg_cycles);
                        endcase
                    end else begin
                        case (op)
                            3'd0:
                                if (rs1[7:0] >= CFG_BASE0 && rs1[7:0] < CFG_PARAM0 + 8'd8) begin
                                    if (!fetch_busy) begin  // fetch unit registers: not while a list runs
                                        reg_we  <= 1;
                                        reg_idx <= rs1[4:0] - CFG_BASE0[4:0];
                                        reg_val <= rs2;
                                        respond(0, 0);
                                    end
                                end else begin
                                case (rs1[7:0])
                                    CFG_LD_ROWS:      ld_rows  <= rs2[15:0];
                                    CFG_LD_ROW_BYTES: ld_rb    <= rs2[15:0];
                                    CFG_LD_PITCH:     ld_pitch <= rs2;
                                    CFG_LD_MODE:      ld_mode  <= rs2[1:0];
                                    CFG_ST_ROWS:      st_rows  <= rs2[15:0];
                                    CFG_ST_ROW_BYTES: st_rb    <= rs2[15:0];
                                    CFG_ST_PITCH:     st_pitch <= rs2;
                                    CFG_EX_REPEAT:    ex_rep   <= rs2[11:0];
                                    CFG_EX_B_STEP:    ex_bstep <= rs2[15:0];
                                    CFG_EX_C_STEP:    ex_cstep <= rs2[15:0];
                                    CFG_EX_C_ROW:     ex_crow  <= rs2[15:0];
                                    default: ;
                                endcase
                                respond(0, 0);
                                end
                            3'd1, 3'd2, 3'd3:
                                if (q_valid && q_ready) begin
                                    q_valid <= 0;
                                    respond(0, 0);
                                end else if (sched_err) begin
                                    q_valid <= 0;           // halted: drop the command
                                    respond(0, 0);
                                end else if (!fetch_busy)   // after a running list
                                    q_valid <= 1;
                            3'd4:                           // mat_fence (also waits for a list)
                                if ((fence_ok && !fetch_busy) || sched_err) respond(1, ext_status);
                            3'd6:                           // mat_submit(list, count)
                                if (sched_err)
                                    respond(0, 0);          // halted: dropped like any command
                                else if (!fetch_busy) begin // one list at a time
                                    submit       <= 1;
                                    submit_addr  <= rs1;
                                    submit_count <= rs2;
                                    respond(0, 0);
                                end
                            3'd7: begin                     // mat_notify: raise the host interrupt
                                notify <= 1;
                                respond(0, 0);
                            end
                            default:                        // mat_perf
                                if (rs1[31]) begin           // control: rs2[0] clear, rs2[1] enable
                                    perf_ctl_we <= 1;
                                    perf_ctl    <= rs2[1:0];
                                    respond(1, NCNT);
                                end else                     // read counter rs1[4:0]
                                    respond(1, perf_rdata);
                        endcase
                    end
                S_DONE:
                    if (!pcpi_valid) state <= S_IDLE;
                default: state <= S_IDLE;
            endcase
        end
    end

    assign perf_ev = {state == S_EXEC && grp == 2'd1 && op == 3'd4 && !(fence_ok || sched_err),
                      state == S_EXEC && q_valid && !q_ready};
endmodule
