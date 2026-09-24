// D x D output-stationary systolic array with shadow accumulators
// (docs/double_buffer_design.md §4).
//
// a_in[8g +: 8] enters row g, b_in[8j +: 8] enters column j; a moves right,
// b moves down one PE per enabled cycle. The caller skews the operands so
// PE(i,j) sees A[i][k] and B[k][j] together and accumulates C[i][j].
//
//   swap   : active -> shadow, active and pass registers cleared (1 cycle,
//            after the last product of a command; the EX engine also holds
//            it during reset to clear the array)
//   shift  : shadows move up one row; row 0 is the drain output, so D shifts
//            deliver C rows 0 .. D-1 on drain_row while the next command
//            already accumulates in the active registers.
//
// Columns 0 .. DSP_COLS-1 use DSP48 MACs, the rest LUT multipliers.
`timescale 1ns / 1ps

(* use_dsp = "yes" *)
module sa_pe_dsp (
    input  wire               clk,
    input  wire               en,
    input  wire               swap,
    input  wire               shift,
    input  wire signed [7:0]  a_in,
    input  wire signed [7:0]  b_in,
    output reg  signed [7:0]  a_out,
    output reg  signed [7:0]  b_out,
    input  wire        [31:0] sh_in,
    output reg         [31:0] sh_out
);
    reg signed [31:0] acc;
    always @(posedge clk) begin
        if (swap) begin
            acc   <= 0;
            a_out <= 0;
            b_out <= 0;
        end else if (en) begin
            acc   <= acc + a_in * b_in;
            a_out <= a_in;
            b_out <= b_in;
        end
        if (swap)       sh_out <= acc;
        else if (shift) sh_out <= sh_in;
    end
endmodule

(* use_dsp = "no" *)
module sa_pe_lut (
    input  wire               clk,
    input  wire               en,
    input  wire               swap,
    input  wire               shift,
    input  wire signed [7:0]  a_in,
    input  wire signed [7:0]  b_in,
    output reg  signed [7:0]  a_out,
    output reg  signed [7:0]  b_out,
    input  wire        [31:0] sh_in,
    output reg         [31:0] sh_out
);
    reg signed [31:0] acc;
    always @(posedge clk) begin
        if (swap) begin
            acc   <= 0;
            a_out <= 0;
            b_out <= 0;
        end else if (en) begin
            acc   <= acc + a_in * b_in;
            a_out <= a_in;
            b_out <= b_in;
        end
        if (swap)       sh_out <= acc;
        else if (shift) sh_out <= sh_in;
    end
endmodule

module sa_array #(
    parameter integer D        = 8,
    parameter integer DSP_COLS = 8
) (
    input  wire              clk,
    input  wire              en,
    input  wire              swap,
    input  wire              shift,
    input  wire [8*D-1:0]    a_in,
    input  wire [8*D-1:0]    b_in,
    output wire [32*D-1:0]   drain_row      // [32j +: 32] = C[row][j]
);
    wire [8*D*(D+1)-1:0]  ah;   // ah(i,j): a entering PE(i,j), j = 0..D
    wire [8*(D+1)*D-1:0]  bv;   // bv(i,j): b entering PE(i,j), i = 0..D
    wire [32*(D+1)*D-1:0] sh;   // sh(i,j): shadow of PE(i,j); row D = 0

    genvar i, j;
    generate
        for (i = 0; i < D; i = i + 1) begin : edge_in
            assign ah[8*(i*(D+1)) +: 8] = a_in[8*i +: 8];
            assign bv[8*i +: 8]         = b_in[8*i +: 8];
            assign sh[32*(D*D + i) +: 32] = 32'd0;
            assign drain_row[32*i +: 32]  = sh[32*i +: 32];
        end
        for (i = 0; i < D; i = i + 1) begin : row
            for (j = 0; j < D; j = j + 1) begin : col
                if (j < DSP_COLS) begin : dsp
                    sa_pe_dsp pe (
                        .clk(clk), .en(en), .swap(swap), .shift(shift),
                        .a_in (ah[8*(i*(D+1)+j)   +: 8]), .b_in (bv[8*(i*D+j)     +: 8]),
                        .a_out(ah[8*(i*(D+1)+j+1) +: 8]), .b_out(bv[8*((i+1)*D+j) +: 8]),
                        .sh_in(sh[32*((i+1)*D+j) +: 32]), .sh_out(sh[32*(i*D+j) +: 32]));
                end else begin : lut
                    sa_pe_lut pe (
                        .clk(clk), .en(en), .swap(swap), .shift(shift),
                        .a_in (ah[8*(i*(D+1)+j)   +: 8]), .b_in (bv[8*(i*D+j)     +: 8]),
                        .a_out(ah[8*(i*(D+1)+j+1) +: 8]), .b_out(bv[8*((i+1)*D+j) +: 8]),
                        .sh_in(sh[32*((i+1)*D+j) +: 32]), .sh_out(sh[32*(i*D+j) +: 32]));
                end
            end
        end
    endgenerate
endmodule
