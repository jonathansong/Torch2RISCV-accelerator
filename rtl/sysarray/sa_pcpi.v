// PCPI front end: decodes the custom-0 matrix instructions from PicoRV32
// (docs/custom_isa_encoding.md, docs/double_buffer_design.md §8).
//
//   funct7 = 0 (Phase 4, kept): mat_trigger / mat_status / mat_reset /
//              mat_wait / mat_cycles -> legacy block
//   funct7 = 1: mat_cfg (0), mat_load (1), mat_store (2), mat_exec (3),
//              mat_fence (4) -> command queue / scheduler
// Load/store/exec capture the current configuration into the packet, so
// software may reconfigure right after issuing. With a sticky error set,
// new commands are dropped (the core never blocks on a halted queue).
// Unclaimed encodings are not answered: the core raises its illegal-
// instruction trap after the PCPI timeout.
`timescale 1ns / 1ps
`include "sa_macros.vh"

module sa_pcpi (
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
    input  wire                  leg_reset_done
);
    `include "sa_defs.vh"

    wire [6:0] opcode = pcpi_insn[6:0];
    wire [2:0] funct3 = pcpi_insn[14:12];
    wire [6:0] funct7 = pcpi_insn[31:25];
    wire       ours   = opcode == 7'b0001011 && funct3 <= 3'd4 &&
                        (funct7 == 7'd0 || funct7 == 7'd1);

    assign pcpi_wait = pcpi_valid && ours;

    // configuration (mat_cfg)
    reg [15:0] ld_rows, ld_rb, st_rows, st_rb;
    reg [31:0] ld_pitch, st_pitch;
    reg [1:0]  ld_mode;

    localparam [1:0] S_IDLE = 0, S_EXEC = 1, S_DONE = 2;
    reg [1:0]  state;
    reg        grp;          // 0: funct7 = 0, 1: funct7 = 1
    reg [2:0]  op;
    reg [31:0] rs1, rs2;

    task respond(input wr, input [31:0] rd);
        begin
            pcpi_ready <= 1; pcpi_wr <= wr; pcpi_rd <= rd; state <= S_DONE;
        end
    endtask

    always @(posedge clk) begin
        pcpi_ready <= 0;
        pcpi_wr    <= 0;
        if (!resetn) begin
            state       <= S_IDLE;
            q_valid     <= 0;
            leg_trigger <= 0;
            leg_reset   <= 0;
            fence_mask  <= 0;
            ld_rows <= 1; ld_rb <= 8; ld_pitch <= 8; ld_mode <= 0;
            st_rows <= 1; st_rb <= 8; st_pitch <= 8;
        end else begin
            case (state)
                S_IDLE:
                    if (pcpi_valid && ours) begin
                        grp   <= funct7[0];
                        op    <= funct3;
                        rs1   <= pcpi_rs1;
                        rs2   <= pcpi_rs2;
                        state <= S_EXEC;
                        // build the packet now, from the configuration current at issue
                        q_pkt <= 0;
                        if (funct7[0] && funct3 == 3'd1)
                            q_pkt <= {1'b0, ld_mode, ld_pitch, ld_rb, ld_rows, pcpi_rs2, pcpi_rs1, CMD_LD};
                        if (funct7[0] && funct3 == 3'd2)
                            q_pkt <= {1'b0, 2'b00, st_pitch, st_rb, st_rows, pcpi_rs2, pcpi_rs1, CMD_ST};
                        if (funct7[0] && funct3 == 3'd3)       // {acc, Kt, C} {B, A}
                            q_pkt <= {70'd0, pcpi_rs2[28], pcpi_rs2[27:16], pcpi_rs2[15:0],
                                      pcpi_rs1[31:16], pcpi_rs1[15:0], CMD_EX};
                        fence_mask <= pcpi_rs1[3:0];
                    end
                S_EXEC:
                    if (!grp) begin
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
                            3'd0: begin
                                case (rs1[7:0])
                                    CFG_LD_ROWS:      ld_rows  <= rs2[15:0];
                                    CFG_LD_ROW_BYTES: ld_rb    <= rs2[15:0];
                                    CFG_LD_PITCH:     ld_pitch <= rs2;
                                    CFG_LD_MODE:      ld_mode  <= rs2[1:0];
                                    CFG_ST_ROWS:      st_rows  <= rs2[15:0];
                                    CFG_ST_ROW_BYTES: st_rb    <= rs2[15:0];
                                    CFG_ST_PITCH:     st_pitch <= rs2;
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
                                end else
                                    q_valid <= 1;
                            default:                        // mat_fence
                                if (fence_ok || sched_err) respond(1, ext_status);
                        endcase
                    end
                S_DONE:
                    if (!pcpi_valid) state <= S_IDLE;
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
