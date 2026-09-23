// Output-stationary systolic array: N x N PEs, int8 x int8 -> int32.
//
// Row i of A enters PE(i,0) from the left, column j of B enters PE(0,j)
// from the top; a moves right and b moves down one PE per cycle. The
// caller skews the inputs (row i / column j delayed by i / j cycles), so
// PE(i,j) sees A[i][k] and B[k][j] in the same cycle and accumulates
// C[i][j] = sum_k A[i][k] * B[k][j] in place.
`timescale 1ns / 1ps

// use_dsp: without it Vivado maps these 8x8 multipliers to LUTs; one
// DSP48E1 per PE absorbs the multiply-accumulate and the acc register.
(* use_dsp = "yes" *)
module systolic_pe (
    input  wire               clk,
    input  wire               clear,
    input  wire               en,
    input  wire signed [7:0]  a_in,
    input  wire signed [7:0]  b_in,
    output reg  signed [7:0]  a_out,
    output reg  signed [7:0]  b_out,
    output reg  signed [31:0] acc
);
    always @(posedge clk) begin
        if (clear) begin
            a_out <= 0;
            b_out <= 0;
            acc   <= 0;
        end else if (en) begin
            a_out <= a_in;
            b_out <= b_in;
            acc   <= acc + a_in * b_in;
        end
    end
endmodule

module systolic_array #(
    parameter integer N = 8
) (
    input  wire              clk,
    input  wire              clear,
    input  wire              en,
    input  wire [8*N-1:0]    a_in,   // a_in[8*i +: 8] feeds row i
    input  wire [8*N-1:0]    b_in,   // b_in[8*j +: 8] feeds column j
    output wire [32*N*N-1:0] acc     // acc[32*(i*N+j) +: 32] = C[i][j]
);
    // ah(i,j): a entering PE(i,j), j = 0..N;  bv(i,j): b entering PE(i,j), i = 0..N
    wire [8*N*(N+1)-1:0] ah;
    wire [8*(N+1)*N-1:0] bv;

    genvar i, j;
    generate
        for (i = 0; i < N; i = i + 1) begin : edge_in
            assign ah[8*(i*(N+1)) +: 8] = a_in[8*i +: 8];
            assign bv[8*i +: 8]         = b_in[8*i +: 8];
        end
        for (i = 0; i < N; i = i + 1) begin : row
            for (j = 0; j < N; j = j + 1) begin : col
                systolic_pe pe (
                    .clk   (clk),
                    .clear (clear),
                    .en    (en),
                    .a_in  (ah[8*(i*(N+1)+j)     +: 8]),
                    .b_in  (bv[8*(i*N+j)         +: 8]),
                    .a_out (ah[8*(i*(N+1)+j+1)   +: 8]),
                    .b_out (bv[8*((i+1)*N+j)     +: 8]),
                    .acc   (acc[32*(i*N+j)       +: 32])
                );
            end
        end
    endgenerate
endmodule
