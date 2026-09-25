// EX engine: runs mat_exec commands on the K-streaming array
// (docs/double_buffer_design.md §4).
//
// Command: A strip at SPAD_A word a (k-tile major: word w*D+g = A[g][wD..wD+D-1]),
// B strip at SPAD_B word b (word k = B[k][0..D-1]), Kt k-tiles, C tile at
// ACC words c .. c+D-1, optional accumulate into the existing ACC contents.
//
// Streaming: at step t the array needs A[g][t-g] in row g and B[t-j][j] in
// column j. With the k-tile-major layout exactly one row needs a new A word
// per step - word a+t, for row t mod D - and column j needs byte j of B word
// t-j, so both SPADs are read sequentially, one word per cycle. Reads are
// issued one cycle ahead (BRAM latency). A command streams for
// K + 2(D-1) steps (K = Kt*D), then swaps into the shadow accumulators; the
// drain (D cycles, 2D with accumulate) runs while the next command streams.
//
// M2: one command computes `repeat` output tiles with the same A strip:
// tile r reads the B strip at b + r*bstep and writes its row i to ACC word
// c + r*cstep + i*crow (cstep = 1, crow = N/D lays a row of C tiles out
// row-major, so one mat_store writes it back). Tiles stream back to back
// like separate commands. `done` pulses once per command, after its last
// tile is written, in command order.
`timescale 1ns / 1ps

module sa_ex #(
    parameter integer D        = 8,
    parameter integer DSP_COLS = 8,
    parameter integer SPAD_AW  = 14,
    parameter integer ACC_AW   = 13
) (
    input  wire                 clk,
    input  wire                 resetn,

    input  wire                 cmd_valid,
    output wire                 cmd_ready,
    input  wire [15:0]          cmd_a,
    input  wire [15:0]          cmd_b,
    input  wire [15:0]          cmd_c,
    input  wire [11:0]          cmd_kt,
    input  wire                 cmd_acc,
    input  wire [11:0]          cmd_rep,       // repeat - 1
    input  wire [15:0]          cmd_bstep,
    input  wire [15:0]          cmd_cstep,
    input  wire [15:0]          cmd_crow,      // 0 means 1
    output reg                  done,
    output wire                 busy,

    output wire                 sa_en,         // SPAD_A read (side B)
    output wire [SPAD_AW-1:0]   sa_addr,
    input  wire [8*D-1:0]       sa_dout,
    output wire                 sb_en,         // SPAD_B read (side B)
    output wire [SPAD_AW-1:0]   sb_addr,
    input  wire [8*D-1:0]       sb_dout,

    output reg                  acc_en,        // ACC (side A)
    output reg  [4*D-1:0]       acc_we,
    output reg  [ACC_AW-1:0]    acc_addr,
    output reg  [32*D-1:0]      acc_din,
    input  wire [32*D-1:0]      acc_dout,

    // performance events: {tile swap, waiting for the previous drain,
    //                      step with new K data, array step}
    output wire [3:0]           perf_ev
);
    localparam integer LD = $clog2(D);

    // ------------------------------------------------------------ stream
    reg         streaming;
    reg  [17:0] c;             // cycle in the command: read word c, step t = c-1
    reg  [17:0] k_len;         // K = Kt*D
    reg  [17:0] t_end;         // K + 2(D-1) + 1: swap cycle
    reg  [15:0] a_base, b_base, c_base;
    reg         acc_mode;
    reg  [11:0] rep_left;      // tiles still to start after the current one
    reg  [15:0] bstep, cstep, crow;

    reg  [8*D-1:0] rowreg [0:D-1];     // current A word of each row
    reg  [8*D-1:0] bsr    [0:D-2];     // B words t-1 .. t-(D-1)

    reg         drain_active;
    wire        swap_now = streaming && c == t_end && !drain_active;

    assign cmd_ready = !streaming && resetn;
    assign busy      = streaming || drain_active;

    assign sa_en   = streaming && c < k_len;
    assign sb_en   = sa_en;
    assign sa_addr = a_base + c;
    assign sb_addr = b_base + c;

    wire [17:0] t        = c - 18'd1;
    wire        step     = streaming && c != 0 && c != t_end;   // array enable
    wire        live_new = t < k_len;                          // this step reads a new word
    wire [LD-1:0] t_low  = t[LD-1:0];

    wire [8*D-1:0] a_feed, b_feed;
    genvar g;
    generate
        for (g = 0; g < D; g = g + 1) begin : feed
            wire          is_new = t_low == g;
            wire [LD-1:0] kk     = t[LD-1:0] - g;               // (t - g) mod D
            assign a_feed[8*g +: 8] = is_new ? (live_new ? sa_dout[7:0] : 8'd0)
                                             : rowreg[g][8*kk +: 8];
            if (g == 0) begin : b0
                assign b_feed[7:0] = live_new ? sb_dout[7:0] : 8'd0;
            end else begin : bj
                assign b_feed[8*g +: 8] = bsr[g-1][8*g +: 8];
            end
        end
    endgenerate

    integer r;
    always @(posedge clk) begin
        if (!resetn) begin
            streaming <= 0;
        end else if (cmd_valid && cmd_ready) begin
            streaming <= 1;
            c         <= 0;
            a_base    <= cmd_a;
            b_base    <= cmd_b;
            c_base    <= cmd_c;
            acc_mode  <= cmd_acc;
            k_len     <= cmd_kt * D;
            t_end     <= cmd_kt * D + 2 * (D - 1) + 1;
            rep_left  <= cmd_rep;
            bstep     <= cmd_bstep;
            cstep     <= cmd_cstep;
            crow      <= cmd_crow == 0 ? 16'd1 : cmd_crow;
            for (r = 0; r < D; r = r + 1)     rowreg[r] <= 0;
            for (r = 0; r < D - 1; r = r + 1) bsr[r]    <= 0;
        end else if (streaming) begin
            if (step) begin
                for (r = 0; r < D; r = r + 1)
                    if (t_low == r) rowreg[r] <= live_new ? sa_dout : {8*D{1'b0}};
                bsr[0] <= live_new ? sb_dout : {8*D{1'b0}};
                for (r = 1; r < D - 1; r = r + 1) bsr[r] <= bsr[r-1];
            end
            if (c != t_end)
                c <= c + 1;
            else if (swap_now) begin     // (waits here while the previous drain runs)
                if (rep_left != 0) begin // next tile of this command, same A strip
                    rep_left <= rep_left - 1;
                    c        <= 0;
                    b_base   <= b_base + bstep;
                    c_base   <= c_base + cstep;
                    for (r = 0; r < D; r = r + 1)     rowreg[r] <= 0;
                    for (r = 0; r < D - 1; r = r + 1) bsr[r]    <= 0;
                end else
                    streaming <= 0;
            end
        end
    end

    // ------------------------------------------------------------- array
    reg         shift;
    wire [32*D-1:0] drain_row;

    sa_array #(.D(D), .DSP_COLS(DSP_COLS)) array (
        .clk       (clk),
        .en        (step),
        .swap      (swap_now || !resetn),
        .shift     (shift),
        .a_in      (a_feed),
        .b_in      (b_feed),
        .drain_row (drain_row)
    );

    // ------------------------------------------------------------- drain
    reg [LD:0]   drow;          // row being drained
    reg          dphase;        // accumulate: 0 = read old, 1 = write sum
    reg [15:0]   d_addr;        // ACC word of the row being drained
    reg [15:0]   d_crow;
    reg          d_acc;
    reg          d_last;        // last tile of its command: pulse done

    integer e;
    always @* begin
        acc_en   = 0;
        acc_we   = 0;
        acc_addr = d_addr;
        acc_din  = drain_row;
        shift    = 0;
        if (drain_active) begin
            acc_en = 1;
            if (!d_acc || dphase) begin
                acc_we = {4*D{1'b1}};
                shift  = 1;
                if (d_acc)
                    for (e = 0; e < D; e = e + 1)
                        acc_din[32*e +: 32] = drain_row[32*e +: 32] + acc_dout[32*e +: 32];
            end
        end
    end

    always @(posedge clk) begin
        done <= 0;
        if (!resetn) begin
            drain_active <= 0;
        end else if (swap_now) begin
            drain_active <= 1;
            drow         <= 0;
            dphase       <= 0;
            d_addr       <= c_base;
            d_crow       <= crow;
            d_acc        <= acc_mode;
            d_last       <= rep_left == 0;
        end else if (drain_active) begin
            if (d_acc && !dphase)
                dphase <= 1;
            else begin
                dphase <= 0;
                drow   <= drow + 1;
                d_addr <= d_addr + d_crow;
                if (drow == D - 1) begin
                    drain_active <= 0;
                    done         <= d_last;
                end
            end
        end
    end

    assign perf_ev = {swap_now, streaming && c == t_end && drain_active, step && live_new, step};
endmodule
