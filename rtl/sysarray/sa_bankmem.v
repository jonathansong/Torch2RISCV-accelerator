// Double-banked true-dual-port memory with byte write enables.
//
// DEPTH words of W bits, split into two banks (bank = MSB of the word index);
// each bank is its own BRAM set with ports A and B, so accesses to different
// banks never contend. Each side (A, B) has NA / NB requesters; per bank and
// side at most one requester may be active in a cycle (the scoreboard
// guarantees it; the simulation checks it). Read data has one cycle of
// latency and is returned to the requester that issued the read.
`timescale 1ns / 1ps

module sa_bankmem #(
    parameter integer W     = 64,
    parameter integer DEPTH = 16384,
    parameter integer NA    = 1,
    parameter integer NB    = 1
) (
    input  wire                           clk,

    input  wire [NA-1:0]                  a_en,
    input  wire [NA*(W/8)-1:0]            a_we,
    input  wire [NA*$clog2(DEPTH)-1:0]    a_addr,
    input  wire [NA*W-1:0]                a_din,
    output wire [NA*W-1:0]                a_dout,

    input  wire [NB-1:0]                  b_en,
    input  wire [NB*(W/8)-1:0]            b_we,
    input  wire [NB*$clog2(DEPTH)-1:0]    b_addr,
    input  wire [NB*W-1:0]                b_din,
    output wire [NB*W-1:0]                b_dout
);
    localparam integer AW  = $clog2(DEPTH);
    localparam integer BAW = AW - 1;           // address inside a bank
    localparam integer BW  = W / 8;

    genvar k, r;
    generate
        for (k = 0; k < 2; k = k + 1) begin : bank
            // ---- side A: pick the requester addressing this bank
            reg           pa_en;
            reg [BW-1:0]  pa_we;
            reg [BAW-1:0] pa_addr;
            reg [W-1:0]   pa_din;
            integer i;
            always @* begin
                pa_en = 0; pa_we = 0; pa_addr = 0; pa_din = 0;
                for (i = NA - 1; i >= 0; i = i - 1)
                    if (a_en[i] && a_addr[AW*i + AW - 1] == k) begin
                        pa_en   = 1;
                        pa_we   = a_we[BW*i +: BW];
                        pa_addr = a_addr[AW*i +: BAW];
                        pa_din  = a_din[W*i +: W];
                    end
            end
            // ---- side B
            reg           pb_en;
            reg [BW-1:0]  pb_we;
            reg [BAW-1:0] pb_addr;
            reg [W-1:0]   pb_din;
            integer j;
            always @* begin
                pb_en = 0; pb_we = 0; pb_addr = 0; pb_din = 0;
                for (j = NB - 1; j >= 0; j = j - 1)
                    if (b_en[j] && b_addr[AW*j + AW - 1] == k) begin
                        pb_en   = 1;
                        pb_we   = b_we[BW*j +: BW];
                        pb_addr = b_addr[AW*j +: BAW];
                        pb_din  = b_din[W*j +: W];
                    end
            end

            wire [W-1:0] qa, qb;
            sa_tdpram #(.NUM_COL(BW), .COL_WIDTH(8), .ADDR_WIDTH(BAW)) ram (
                .clk(clk),
                .enaA(pa_en), .weA(pa_we), .addrA(pa_addr), .dinA(pa_din), .doutA(qa),
                .enaB(pb_en), .weB(pb_we), .addrB(pb_addr), .dinB(pb_din), .doutB(qb));
        end

        // ---- return read data to the requester, by the bank it addressed
        for (r = 0; r < NA; r = r + 1) begin : ret_a
            reg sel;
            always @(posedge clk) if (a_en[r]) sel <= a_addr[AW*r + AW - 1];
            assign a_dout[W*r +: W] = sel ? bank[1].qa : bank[0].qa;
        end
        for (r = 0; r < NB; r = r + 1) begin : ret_b
            reg sel;
            always @(posedge clk) if (b_en[r]) sel <= b_addr[AW*r + AW - 1];
            assign b_dout[W*r +: W] = sel ? bank[1].qb : bank[0].qb;
        end
    endgenerate

`ifndef SYNTHESIS
    // two requesters on the same bank and side in one cycle = scoreboard bug
    integer p, q;
    always @(posedge clk) begin
        for (p = 0; p < NA; p = p + 1)
            for (q = p + 1; q < NA; q = q + 1)
                if (a_en[p] && a_en[q] && a_addr[AW*p + AW - 1] == a_addr[AW*q + AW - 1])
                    $display("FATAL: sa_bankmem %m side A requesters %0d/%0d collide on a bank", p, q);
        for (p = 0; p < NB; p = p + 1)
            for (q = p + 1; q < NB; q = q + 1)
                if (b_en[p] && b_en[q] && b_addr[AW*p + AW - 1] == b_addr[AW*q + AW - 1])
                    $display("FATAL: sa_bankmem %m side B requesters %0d/%0d collide on a bank", p, q);
    end
`endif
endmodule
