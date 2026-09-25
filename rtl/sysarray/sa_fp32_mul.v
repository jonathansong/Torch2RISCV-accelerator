// fp32 multiplier for the L2 vector engine (docs/llm_inference_plan.md §3.2).
//
// Same rules as sa_fp32_add.v (round to nearest even, flush to zero,
// overflow -> +-inf, canonical NaN, 0 * inf -> NaN, signed zeros); bit-exact
// with llm/fp32.py mul().
//
// Pipeline, 4 stages (LAT = 4): 1 unpack + specials; 2-3 the 24 x 24
// significand product (DSP48, registered twice); 4 normalize, round, pack.
`timescale 1ns / 1ps

module sa_fp32_mul #(
    parameter integer TW = 1
) (
    input  wire          clk,
    input  wire          in_v,
    input  wire [31:0]   a,
    input  wire [31:0]   b,
    input  wire [TW-1:0] in_tag,
    output reg           out_v,
    output reg  [31:0]   y,
    output reg  [TW-1:0] out_tag
);
    localparam [31:0] QNAN = 32'h7FC0_0000;

    // ------------------------------------------------------------ stage 1
    wire [7:0] ea = a[30:23], eb = b[30:23];
    wire       za = ea == 0, zb = eb == 0;
    wire       ia = ea == 8'hFF && a[22:0] == 0, ib = eb == 8'hFF && b[22:0] == 0;
    wire       na = ea == 8'hFF && a[22:0] != 0, nb = eb == 8'hFF && b[22:0] != 0;
    wire       s  = a[31] ^ b[31];

    reg          v1;
    reg [TW-1:0] t1;
    reg          s1, spec1;
    reg [31:0]   spec_y1;
    reg [23:0]   ma1, mb1;
    reg signed [10:0] e1;
    always @(posedge clk) begin
        v1    <= in_v;
        t1    <= in_tag;
        s1    <= s;
        ma1   <= {1'b1, a[22:0]};
        mb1   <= {1'b1, b[22:0]};
        e1    <= $signed({3'd0, ea}) + $signed({3'd0, eb}) - 11'sd127;
        spec1 <= na || nb || ia || ib || za || zb;
        if (na || nb || ((ia || ib) && (za || zb)))
            spec_y1 <= QNAN;
        else if (ia || ib)
            spec_y1 <= {s, 8'hFF, 23'd0};
        else
            spec_y1 <= {s, 31'd0};                       // a zero operand (FTZ)
    end

    // --------------------------------------------------------- stages 2-3
    (* use_dsp = "yes" *) reg [47:0] p2, p3;
    reg          v2, v3;
    reg [TW-1:0] t2, t3;
    reg          s2, s3, spec2, spec3;
    reg [31:0]   spec_y2, spec_y3;
    reg signed [10:0] e2, e3;
    always @(posedge clk) begin
        p2 <= ma1 * mb1;
        v2 <= v1; t2 <= t1; s2 <= s1; spec2 <= spec1; spec_y2 <= spec_y1; e2 <= e1;
        p3 <= p2;
        v3 <= v2; t3 <= t2; s3 <= s2; spec3 <= spec2; spec_y3 <= spec_y2; e3 <= e2;
    end

    // ------------------------------------------------------------ stage 4
    wire        hi   = p3[47];                           // product in [2^47, 2^48)
    wire [23:0] q    = hi ? p3[47:24] : p3[46:23];
    wire        g    = hi ? p3[23] : p3[22];             // first dropped bit
    wire        st   = hi ? p3[22:0] != 0 : p3[21:0] != 0;
    wire        up   = g && (st || q[0]);
    wire [24:0] qr   = {1'b0, q} + {24'd0, up};
    wire        ovf  = qr[24];
    wire [22:0] frac = ovf ? qr[23:1] : qr[22:0];
    wire signed [10:0] ef = e3 + (hi ? 11'sd1 : 11'sd0) + (ovf ? 11'sd1 : 11'sd0);

    always @(posedge clk) begin
        out_v   <= v3;
        out_tag <= t3;
        if (spec3)
            y <= spec_y3;
        else if (ef < 11'sd1)
            y <= {s3, 31'd0};
        else if (ef > 11'sd254)
            y <= {s3, 8'hFF, 23'd0};
        else
            y <= {s3, ef[7:0], frac};
    end
endmodule
