// True dual-port RAM with byte-wide write enables, read-first, 1-cycle read
// latency: the Vivado inference template of UG901 ("byte-write true dual
// port"), kept literal so it maps to RAMB36 with byte write enables.
`timescale 1ns / 1ps

module sa_tdpram #(
    parameter integer NUM_COL    = 8,
    parameter integer COL_WIDTH  = 8,
    parameter integer ADDR_WIDTH = 13,
    parameter integer DATA_WIDTH = NUM_COL * COL_WIDTH
) (
    input  wire                  clk,
    input  wire                  enaA,
    input  wire [NUM_COL-1:0]    weA,
    input  wire [ADDR_WIDTH-1:0] addrA,
    input  wire [DATA_WIDTH-1:0] dinA,
    output reg  [DATA_WIDTH-1:0] doutA,
    input  wire                  enaB,
    input  wire [NUM_COL-1:0]    weB,
    input  wire [ADDR_WIDTH-1:0] addrB,
    input  wire [DATA_WIDTH-1:0] dinB,
    output reg  [DATA_WIDTH-1:0] doutB
);
    (* ram_style = "block" *) reg [DATA_WIDTH-1:0] ram_block [(2**ADDR_WIDTH)-1:0];

    integer i;
    always @(posedge clk) begin
        if (enaA) begin
            for (i = 0; i < NUM_COL; i = i + 1)
                if (weA[i])
                    ram_block[addrA][i*COL_WIDTH +: COL_WIDTH] <= dinA[i*COL_WIDTH +: COL_WIDTH];
            doutA <= ram_block[addrA];
        end
    end

    integer j;
    always @(posedge clk) begin
        if (enaB) begin
            for (j = 0; j < NUM_COL; j = j + 1)
                if (weB[j])
                    ram_block[addrB][j*COL_WIDTH +: COL_WIDTH] <= dinB[j*COL_WIDTH +: COL_WIDTH];
            doutB <= ram_block[addrB];
        end
    end
endmodule
