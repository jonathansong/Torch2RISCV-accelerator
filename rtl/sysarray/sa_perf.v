// Performance counters (docs/perf_counters_and_desc_dma_plan.md, part 1).
//
// NCNT 32-bit event counters. ev[i] is a one-cycle event from an engine,
// the scheduler or the PCPI front end (numbering: PC_* in sa_defs.vh;
// ev[0] = 1 counts cycles). Events are registered once before counting so
// no engine logic lands on a new long path; counts lag by one cycle.
//
// Counting only while `en`; control comes from the PCPI instruction mat_perf
// (port a) or the PERF_CTRL CSR (port b): ctl[0] = clear, ctl[1] = enable.
// Two combinational read ports (PCPI, CSR mirror); both callers register the
// value themselves. PERF = 0 removes the counters: reads return 0.
`timescale 1ns / 1ps

module sa_perf #(
    parameter integer NCNT = 32,
    parameter integer PERF = 1
) (
    input  wire              clk,
    input  wire              resetn,
    input  wire [NCNT-1:0]   ev,

    input  wire              ctl_we_a,      // PCPI mat_perf control
    input  wire [1:0]        ctl_a,
    input  wire              ctl_we_b,      // PERF_CTRL CSR write
    input  wire [1:0]        ctl_b,
    output wire              en,

    input  wire [4:0]        rsel_a,        // PCPI read
    output wire [31:0]       rdata_a,
    input  wire [4:0]        rsel_b,        // CSR mirror read
    output wire [31:0]       rdata_b
);
    generate
        if (PERF != 0) begin : on
            reg              en_r;
            reg  [NCNT-1:0]  ev_q;
            reg  [31:0]      cnt [0:NCNT-1];
            wire             clear = (ctl_we_a && ctl_a[0]) || (ctl_we_b && ctl_b[0]);

            integer i;
            always @(posedge clk) begin
                if (!resetn) begin
                    en_r <= 0;
                    ev_q <= 0;
                    for (i = 0; i < NCNT; i = i + 1) cnt[i] <= 0;
                end else begin
                    ev_q <= ev;
                    if (ctl_we_a)      en_r <= ctl_a[1];      // PCPI wins a same-cycle write
                    else if (ctl_we_b) en_r <= ctl_b[1];
                    for (i = 0; i < NCNT; i = i + 1)
                        if (clear)                 cnt[i] <= 0;
                        else if (en_r && ev_q[i])  cnt[i] <= cnt[i] + 1;
                end
            end

            assign en      = en_r;
            assign rdata_a = rsel_a < NCNT ? cnt[rsel_a] : 32'd0;
            assign rdata_b = rsel_b < NCNT ? cnt[rsel_b] : 32'd0;
        end else begin : off
            assign en      = 1'b0;
            assign rdata_a = 32'd0;
            assign rdata_b = 32'd0;
        end
    endgenerate
endmodule
