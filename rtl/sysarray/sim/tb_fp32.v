// L2 fp32 units against the bit-exact model (sim/gen_fp32_vectors.py ->
// llm/fp32.py): every vector streamed one per cycle, results compared in
// order through the tag.
`timescale 1ns / 1ps

module tb_fp32;
    `include "fp_counts.vh"
    reg clk = 0;
    always #5 clk = ~clk;
    integer errors = 0, i, got_n;

    // vector memories: $fscanf per line
    reg [31:0] add_a [0:N_ADD-1], add_b [0:N_ADD-1], add_y [0:N_ADD-1];
    reg        add_s [0:N_ADD-1];
    reg [31:0] mul_a [0:N_MUL-1], mul_b [0:N_MUL-1], mul_y [0:N_MUL-1];
    reg [31:0] i2f_x [0:N_I2F-1], i2f_y [0:N_I2F-1];
    reg [31:0] f2i_x [0:N_F2I-1], f2i_lo [0:N_F2I-1], f2i_hi [0:N_F2I-1], f2i_y [0:N_F2I-1];
    reg        f2i_f [0:N_F2I-1];
    integer fd, r;
    reg [31:0] t0, t1, t2, t3, t4;
    initial begin
        fd = $fopen("fp_add.hex", "r");
        for (i = 0; i < N_ADD; i = i + 1) begin r = $fscanf(fd, "%h %h %h %h\n", t0, t1, t2, t3);
            add_a[i] = t0; add_b[i] = t1; add_s[i] = t2[0]; add_y[i] = t3; end
        $fclose(fd);
        fd = $fopen("fp_mul.hex", "r");
        for (i = 0; i < N_MUL; i = i + 1) begin r = $fscanf(fd, "%h %h %h\n", t0, t1, t2);
            mul_a[i] = t0; mul_b[i] = t1; mul_y[i] = t2; end
        $fclose(fd);
        fd = $fopen("fp_i2f.hex", "r");
        for (i = 0; i < N_I2F; i = i + 1) begin r = $fscanf(fd, "%h %h\n", t0, t1);
            i2f_x[i] = t0; i2f_y[i] = t1; end
        $fclose(fd);
        fd = $fopen("fp_f2i.hex", "r");
        for (i = 0; i < N_F2I; i = i + 1) begin r = $fscanf(fd, "%h %h %h %h %h\n", t0, t1, t2, t3, t4);
            f2i_x[i] = t0; f2i_f[i] = t1[0]; f2i_lo[i] = t2; f2i_hi[i] = t3; f2i_y[i] = t4; end
        $fclose(fd);
    end

    // ---- units, driven from index counters
    reg         run = 0;
    reg  [31:0] ia = 0, im = 0, ii = 0, io = 0;
    wire        add_v = run && ia < N_ADD, mul_v = run && im < N_MUL;
    wire        i2f_v = run && ii < N_I2F, f2i_v = run && io < N_F2I;
    wire        add_ov, mul_ov, i2f_ov, f2i_ov;
    wire [31:0] add_o, mul_o, i2f_o, f2i_o, add_t, mul_t, i2f_t, f2i_t;
    sa_fp32_add #(.TW(32)) u_add (.clk(clk), .in_v(add_v), .a(add_a[ia]), .b(add_b[ia]), .sub(add_s[ia]),
                                  .in_tag(ia), .out_v(add_ov), .y(add_o), .out_tag(add_t));
    sa_fp32_mul #(.TW(32)) u_mul (.clk(clk), .in_v(mul_v), .a(mul_a[im]), .b(mul_b[im]),
                                  .in_tag(im), .out_v(mul_ov), .y(mul_o), .out_tag(mul_t));
    sa_fp32_i2f #(.TW(32)) u_i2f (.clk(clk), .in_v(i2f_v), .x(i2f_x[ii]), .in_tag(ii),
                                  .out_v(i2f_ov), .y(i2f_o), .out_tag(i2f_t));
    sa_fp32_f2i #(.TW(32)) u_f2i (.clk(clk), .in_v(f2i_v), .x(f2i_x[io]), .floor_mode(f2i_f[io]),
                                  .lo(f2i_lo[io]), .hi(f2i_hi[io]), .in_tag(io),
                                  .out_v(f2i_ov), .y(f2i_o), .out_tag(f2i_t));
    integer n_add = 0, n_mul = 0, n_i2f = 0, n_f2i = 0, e_add = 0, e_mul = 0, e_i2f = 0, e_f2i = 0;
    always @(posedge clk) if (run) begin
        if (add_v) ia <= ia + 1;
        if (mul_v) im <= im + 1;
        if (i2f_v) ii <= ii + 1;
        if (f2i_v) io <= io + 1;
        if (add_ov) begin
            n_add = n_add + 1;
            if (add_o !== add_y[add_t]) begin
                e_add = e_add + 1;
                if (e_add <= 5) $display("TB ERROR add %08x %s %08x = %08x, expected %08x", add_a[add_t],
                                         add_s[add_t] ? "-" : "+", add_b[add_t], add_o, add_y[add_t]);
            end
        end
        if (mul_ov) begin
            n_mul = n_mul + 1;
            if (mul_o !== mul_y[mul_t]) begin
                e_mul = e_mul + 1;
                if (e_mul <= 5) $display("TB ERROR mul %08x * %08x = %08x, expected %08x", mul_a[mul_t],
                                         mul_b[mul_t], mul_o, mul_y[mul_t]);
            end
        end
        if (i2f_ov) begin
            n_i2f = n_i2f + 1;
            if (i2f_o !== i2f_y[i2f_t]) begin
                e_i2f = e_i2f + 1;
                if (e_i2f <= 5) $display("TB ERROR i2f %08x = %08x, expected %08x", i2f_x[i2f_t], i2f_o, i2f_y[i2f_t]);
            end
        end
        if (f2i_ov) begin
            n_f2i = n_f2i + 1;
            if (f2i_o !== f2i_y[f2i_t]) begin
                e_f2i = e_f2i + 1;
                if (e_f2i <= 5) $display("TB ERROR f2i %08x (%s, %0d..%0d) = %0d, expected %0d", f2i_x[f2i_t],
                                         f2i_f[f2i_t] ? "floor" : "rne", $signed(f2i_lo[f2i_t]), $signed(f2i_hi[f2i_t]),
                                         $signed(f2i_o), $signed(f2i_y[f2i_t]));
            end
        end
    end
    initial begin
        repeat (3) @(posedge clk);
        run = 1;
        while (n_add < N_ADD || n_mul < N_MUL || n_i2f < N_I2F || n_f2i < N_F2I) @(posedge clk);
        errors = e_add + e_mul + e_i2f + e_f2i;
        $display("TB %s: fp32 units vs llm/fp32.py: add/sub %0d (%0d errors), mul %0d (%0d), i2f %0d (%0d), f2i %0d (%0d)",
                 errors ? "FAIL" : "PASS", n_add, e_add, n_mul, e_mul, n_i2f, e_i2f, n_f2i, e_f2i);
        $finish;
    end
endmodule
