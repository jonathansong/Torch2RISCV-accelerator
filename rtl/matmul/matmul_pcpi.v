// PCPI front end for matmul_unit: decodes the custom-0 matrix instructions
// issued by PicoRV32 and turns them into commands for the matmul FSM.
// Encoding (docs/custom_isa_encoding.md), R-type, opcode 0x0B, funct7 = 0:
//
//   funct3 0  mat_trigger rs1=descriptor, rs2=C address   stalls only while busy
//   funct3 1  mat_status  rd                              STATUS, no stall
//   funct3 2  mat_reset                                   clear status (waits for idle)
//   funct3 3  mat_wait    rd                              stall until idle, rd = STATUS
//   funct3 4  mat_cycles  rd                              CYCLES of the last job
//
// Any other custom-0 encoding is left unanswered, so the core raises its
// illegal-instruction trap after the 16-cycle PCPI timeout.
`timescale 1ns / 1ps

module matmul_pcpi (
    input  wire        clk,
    input  wire        resetn,

    input  wire        pcpi_valid,
    input  wire [31:0] pcpi_insn,
    input  wire [31:0] pcpi_rs1,
    input  wire [31:0] pcpi_rs2,
    output reg         pcpi_wr,
    output reg  [31:0] pcpi_rd,
    output wire        pcpi_wait,
    output reg         pcpi_ready,

    input  wire        busy,
    input  wire [31:0] status,
    input  wire [31:0] cycles,
    output reg         cmd_valid,     // held until cmd_accept
    output reg  [31:0] cmd_desc,
    output reg  [31:0] cmd_dst,
    input  wire        cmd_accept,
    output reg         cmd_reset      // one-cycle pulse, only issued when idle
);
    localparam [6:0] OPC_CUSTOM0 = 7'b0001011;
    localparam [2:0] F_TRIGGER = 3'd0, F_STATUS = 3'd1, F_RESET = 3'd2,
                     F_WAIT    = 3'd3, F_CYCLES = 3'd4;

    wire [2:0] funct3 = pcpi_insn[14:12];
    wire       ours   = pcpi_insn[6:0] == OPC_CUSTOM0 && pcpi_insn[31:25] == 7'd0 &&
                        funct3 <= F_CYCLES;

    // Keep the core from timing out for as long as we own the instruction.
    assign pcpi_wait = pcpi_valid && ours;

    localparam [1:0] S_IDLE = 2'd0, S_EXEC = 2'd1, S_DONE = 2'd2;
    reg [1:0] state;
    reg [2:0] op;

    task respond(input wr, input [31:0] rd);
        begin
            pcpi_ready <= 1;
            pcpi_wr    <= wr;
            pcpi_rd    <= rd;
            state      <= S_DONE;
        end
    endtask

    always @(posedge clk) begin
        pcpi_ready <= 0;
        pcpi_wr    <= 0;
        cmd_reset  <= 0;
        if (!resetn) begin
            state     <= S_IDLE;
            cmd_valid <= 0;
        end else begin
            case (state)
                S_IDLE:
                    if (pcpi_valid && ours) begin
                        op       <= funct3;
                        cmd_desc <= pcpi_rs1;
                        cmd_dst  <= pcpi_rs2;
                        state    <= S_EXEC;
                    end
                S_EXEC:
                    case (op)
                        F_TRIGGER:
                            if (cmd_valid && cmd_accept) begin
                                cmd_valid <= 0;
                                respond(0, 32'd0);
                            end else
                                cmd_valid <= 1;
                        F_STATUS: respond(1, status);
                        F_RESET:
                            if (!busy) begin
                                cmd_reset <= 1;
                                respond(0, 32'd0);
                            end
                        F_WAIT:   if (!busy) respond(1, status);
                        default:  respond(1, cycles);          // F_CYCLES
                    endcase
                // The core drops pcpi_valid on the edge it sees pcpi_ready;
                // wait for that so one instruction is never executed twice.
                S_DONE:
                    if (!pcpi_valid) state <= S_IDLE;
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
