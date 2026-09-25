// fp32 adder for the L2 vector engine (docs/llm_inference_plan.md §3.2).
//
// IEEE-754 binary32, round to nearest even, flush to zero (subnormal inputs
// are zeros; a result whose exponent after rounding is below the normal
// range is a zero of its sign), overflow -> +-inf, NaN in / inf - inf ->
// canonical NaN 0x7FC00000, x + (-x) = +0, -0 + -0 = -0. Bit-exact with
// llm/fp32.py add(). Subtraction: flip the sign of b (sub = 1).
//
// Pipeline, 4 stages (result LAT = 4 cycles after the operands; one
// operation per cycle; `tag` travels along):
//   1 unpack, specials, order by magnitude, exponent difference
//   2 align the smaller significand (guard, round, sticky), add / subtract
//   3 normalize (leading-one position)
//   4 round to nearest even, renormalize, pack (FTZ / overflow), specials
`timescale 1ns / 1ps

module sa_fp32_add #(
    parameter integer TW = 1                   // tag width
) (
    input  wire          clk,
    input  wire          in_v,
    input  wire [31:0]   a,
    input  wire [31:0]   b,
    input  wire          sub,
    input  wire [TW-1:0] in_tag,
    output reg           out_v,
    output reg  [31:0]   y,
    output reg  [TW-1:0] out_tag
);
    localparam [31:0] QNAN = 32'h7FC0_0000;

    // ------------------------------------------------------------ stage 1
    wire [31:0] bb = b ^ {sub, 31'd0};
    wire        sa = a[31], sb = bb[31];
    wire [7:0]  ea = a[30:23], eb = bb[30:23];
    wire        za = ea == 0, zb = eb == 0;              // FTZ: subnormals are zeros
    wire        ia = ea == 8'hFF && a[22:0] == 0, ib = eb == 8'hFF && bb[22:0] == 0;
    wire        na = ea == 8'hFF && a[22:0] != 0, nb = eb == 8'hFF && bb[22:0] != 0;
    wire [23:0] ma = za ? 24'd0 : {1'b1, a[22:0]};
    wire [23:0] mb = zb ? 24'd0 : {1'b1, bb[22:0]};
    wire        swap = {eb, mb} > {ea, ma};

    reg          v1;
    reg [TW-1:0] t1;
    reg          sx1, same1, zero1;
    reg [7:0]    ex1, d1;
    reg [23:0]   mx1, my1;
    reg          spec1;
    reg [31:0]   spec_y1;
    always @(posedge clk) begin
        v1    <= in_v;
        t1    <= in_tag;
        sx1   <= swap ? sb : sa;
        same1 <= sa == sb;
        ex1   <= swap ? eb : ea;
        mx1   <= swap ? mb : ma;
        my1   <= swap ? ma : mb;
        d1    <= swap ? eb - ea : ea - eb;
        zero1 <= za && zb;
        // results that do not need the datapath
        spec1 <= na || nb || ia || ib || za || zb;
        if (na || nb || (ia && ib && sa != sb))
            spec_y1 <= QNAN;
        else if (ia)
            spec_y1 <= a;
        else if (ib)
            spec_y1 <= bb;
        else if (za && zb)
            spec_y1 <= {sa & sb, 31'd0};
        else if (za)
            spec_y1 <= bb;                               // b is normal: unchanged
        else
            spec_y1 <= a;                                // zb: a unchanged
    end

    // ------------------------------------------------------------ stage 2
    wire [26:0] X = {mx1, 3'b000};
    wire [26:0] Yf = {my1, 3'b000};
    wire [26:0] Ysh = d1 >= 8'd27 ? 27'd0 : Yf >> d1;
    wire        stk = d1 >= 8'd27 ? (my1 != 0) : ((Ysh << d1) != Yf);
    wire [26:0] Y = Ysh | {26'd0, stk};

    reg          v2;
    reg [TW-1:0] t2;
    reg          sx2, spec2;
    reg [31:0]   spec_y2;
    reg [7:0]    ex2;
    reg [27:0]   S2;
    always @(posedge clk) begin
        v2      <= v1;
        t2      <= t1;
        sx2     <= sx1;
        ex2     <= ex1;
        spec2   <= spec1;
        spec_y2 <= spec_y1;
        S2      <= same1 ? {1'b0, X} + {1'b0, Y} : {1'b0, X} - {1'b0, Y};
    end

    // ------------------------------------------------------------ stage 3
    // top = position of the leading one of S2 (0..27)
    reg  [4:0] top;
    integer k;
    always @* begin
        top = 0;
        for (k = 0; k < 28; k = k + 1)
            if (S2[k]) top = k[4:0];
    end
    wire [4:0]  lsh = top < 5'd26 ? 5'd26 - top : 5'd0;
    wire [26:0] N  = top == 5'd27 ? {S2[27:2], S2[1] | S2[0]} : S2[26:0] << lsh;

    reg          v3;
    reg [TW-1:0] t3;
    reg          sx3, spec3, zres3;
    reg [31:0]   spec_y3;
    reg signed [10:0] e3;
    reg [26:0]   N3;
    always @(posedge clk) begin
        v3      <= v2;
        t3      <= t2;
        sx3     <= sx2;
        spec3   <= spec2;
        spec_y3 <= spec_y2;
        zres3   <= S2 == 0;
        N3      <= N;
        e3      <= $signed({3'd0, ex2}) + (top == 5'd27 ? 11'sd1 : -$signed({6'd0, lsh}));
    end

    // ------------------------------------------------------------ stage 4
    wire [23:0] q    = N3[26:3];
    wire [2:0]  grs  = N3[2:0];
    wire        up   = grs > 3'd4 || (grs == 3'd4 && q[0]);
    wire [24:0] qr   = {1'b0, q} + {24'd0, up};
    wire        ovf  = qr[24];
    wire [22:0] frac = ovf ? qr[23:1] : qr[22:0];
    wire signed [10:0] ef = e3 + (ovf ? 11'sd1 : 11'sd0);

    always @(posedge clk) begin
        out_v   <= v3;
        out_tag <= t3;
        if (spec3)
            y <= spec_y3;
        else if (zres3)
            y <= 32'd0;                                  // exact cancellation: +0
        else if (ef < 11'sd1)
            y <= {sx3, 31'd0};                           // FTZ
        else if (ef > 11'sd254)
            y <= {sx3, 8'hFF, 23'd0};                    // overflow
        else
            y <= {sx3, ef[7:0], frac};
    end
endmodule
