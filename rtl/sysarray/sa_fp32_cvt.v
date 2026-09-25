// fp32 conversions for the L2 vector engine, bit-exact with llm/fp32.py:
//   sa_fp32_i2f  int32 -> fp32, round to nearest even        (LAT = 2)
//   sa_fp32_f2i  fp32 -> int32, round to nearest even or floor, then
//                saturate to [lo, hi]; NaN -> 0; subnormal -> 0 (LAT = 2)
`timescale 1ns / 1ps

module sa_fp32_i2f #(
    parameter integer TW = 1
) (
    input  wire          clk,
    input  wire          in_v,
    input  wire [31:0]   x,
    input  wire [TW-1:0] in_tag,
    output reg           out_v,
    output reg  [31:0]   y,
    output reg  [TW-1:0] out_tag
);
    wire        s = x[31];
    wire [31:0] v = s ? -x : x;                          // |x| (2^31 for INT_MIN)
    reg  [4:0]  top;
    integer k;
    always @* begin
        top = 0;
        for (k = 0; k < 32; k = k + 1)
            if (v[k]) top = k[4:0];
    end

    reg          v1, s1, z1;
    reg [TW-1:0] t1;
    reg [4:0]    top1;
    reg [31:0]   n1;                                     // |x| with its leading one at bit 31
    always @(posedge clk) begin
        v1 <= in_v; t1 <= in_tag; s1 <= s; z1 <= x == 0; top1 <= top;
        n1 <= v << (5'd31 - top);
    end

    wire [23:0] q   = n1[31:8];
    wire        g   = n1[7];
    wire        st  = n1[6:0] != 0;
    wire        up  = g && (st || q[0]);
    wire [24:0] qr  = {1'b0, q} + {24'd0, up};
    wire        ovf = qr[24];
    wire [22:0] frac = ovf ? qr[23:1] : qr[22:0];
    wire [7:0]  e    = 8'd127 + {3'd0, top1} + {7'd0, ovf};
    always @(posedge clk) begin
        out_v <= v1; out_tag <= t1;
        y <= z1 ? 32'd0 : {s1, e, frac};
    end
endmodule

module sa_fp32_f2i #(
    parameter integer TW = 1
) (
    input  wire          clk,
    input  wire          in_v,
    input  wire [31:0]   x,
    input  wire          floor_mode,                     // 0: round to nearest even, 1: floor
    input  wire [31:0]   lo,                             // saturation bounds (signed)
    input  wire [31:0]   hi,
    input  wire [TW-1:0] in_tag,
    output reg           out_v,
    output reg  [31:0]   y,
    output reg  [TW-1:0] out_tag
);
    wire        s  = x[31];
    wire [7:0]  e  = x[30:23];
    wire        zero = e == 0;
    wire        nan  = e == 8'hFF && x[22:0] != 0;
    wire        inf  = e == 8'hFF && x[22:0] == 0;
    wire [23:0] m  = {1'b1, x[22:0]};
    // value = m * 2^(e - 150)
    wire        big   = e >= 8'd158;                     // |value| >= 2^31
    wire        left  = e >= 8'd150;
    wire [7:0]  sh    = 8'd150 - e;                      // right shift when !left
    wire [7:0]  shc   = sh > 8'd30 ? 8'd30 : sh;         // m < 2^24: >= 25 leaves 0
    wire [30:0] lmag  = {7'd0, m} << (e - 8'd150);       // e in 150 .. 157
    wire [23:0] q     = shc >= 8'd25 ? 24'd0 : m >> shc;
    wire [54:0] mfull = {m, 31'd0};
    wire [54:0] rem   = (mfull >> shc) & 55'h7FFF_FFFF;   // the dropped bits, left-aligned at bit 30
    // rem[30] = first dropped bit, rem[29:0] the rest (for shc <= 30; larger shifts: m >> shc = 0,
    // all of m is dropped and lies below the half point for sh >= 26)
    wire        half  = sh <= 8'd30 ? rem[30] : 1'b0;
    wire        rest  = sh <= 8'd30 ? rem[29:0] != 0 : 1'b1;
    wire        inexact = half || rest;

    reg          v1, s1, z1, n1, sat1, fl1;
    reg [TW-1:0] t1;
    reg [31:0]   mag1;
    reg          up1, dn1;
    reg [31:0]   lo1, hi1;
    always @(posedge clk) begin
        v1 <= in_v; t1 <= in_tag; s1 <= s; z1 <= zero; n1 <= nan; sat1 <= big || inf;
        lo1 <= lo; hi1 <= hi; fl1 <= floor_mode;
        mag1 <= left ? {1'b0, lmag} : {8'd0, q};
        up1  <= !left && half && (rest || q[0]);          // round to nearest even: magnitude + 1
        dn1  <= !left && inexact;                         // floor of a negative inexact value: - 1
    end

    wire signed [33:0] mag  = {2'b00, mag1} + (!fl1 && up1 ? 34'sd1 : 34'sd0);
    wire signed [33:0] sv   = s1 ? -mag - (fl1 && dn1 ? 34'sd1 : 34'sd0) : mag;
    wire signed [33:0] slo  = {{2{lo1[31]}}, lo1};
    wire signed [33:0] shi  = {{2{hi1[31]}}, hi1};
    wire signed [33:0] satv = s1 ? slo : shi;
    wire signed [33:0] v2   = z1 ? 34'sd0 : sat1 ? satv : sv;     // a zero is clipped too (lo > 0)
    wire signed [33:0] cl   = v2 < slo ? slo : v2 > shi ? shi : v2;
    always @(posedge clk) begin
        out_v <= v1; out_tag <= t1;
        y <= n1 ? 32'd0 : cl[31:0];
    end
endmodule
