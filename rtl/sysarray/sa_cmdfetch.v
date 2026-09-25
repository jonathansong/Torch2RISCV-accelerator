// Descriptor fetch unit (docs/perf_counters_and_desc_dma_plan.md part 2,
// docs/double_buffer_design.md §8.6).
//
// mat_submit(list, count) starts it: 64-byte descriptors are read from DDR
// (one 8-beat burst each, up to 4 in flight, into a 32 x 64-bit LUTRAM FIFO),
// assembled, decoded and fed to the scheduler input in list order, exactly
// like the commands the PCPI decoder builds (the pkt_* functions of
// sa_defs.vh). Control descriptors are handled here:
//   FENCE  stop feeding until the scheduler queue is empty and the engines in
//          the mask (0 = all) are idle
//   JUMP   continue at another 64-byte aligned list address
//   END    record the status value, count a finished list, stop
// A list also stops after `count` descriptors (count 0 = until END).
// Descriptors are prefetched speculatively; after JUMP / END / the count / an
// error the unit drops the bursts still in flight (DRAIN) before it goes on.
//
// Errors (reported to the scheduler's sticky error with engine ENG_FETCH):
// a misaligned list / jump address or an invalid header (opcode, reserved
// bits [31:13]) -> XERR_SHAPE; a read response error -> XERR_RRESP. The unit
// halts until clear_error (mat_reset), then drains and goes idle. It also
// halts when the scheduler reports any other error.
`timescale 1ns / 1ps
`include "sa_macros.vh"

module sa_cmdfetch #(
    parameter integer D = 8                    // VE LEN (elements) -> groups
) (
    input  wire                  clk,
    input  wire                  resetn,

    // mat_submit
    input  wire                  submit,       // one-cycle pulse, only while !busy
    input  wire [31:0]           submit_addr,
    input  wire [31:0]           submit_count, // 0 = until END
    output wire                  busy,
    input  wire [127:0]          bases,        // BASE3 .. BASE0 (mat_cfg keys 11..14)

    // scheduler
    output wire                  out_valid,
    input  wire                  out_ready,
    output wire [`SA_PKT_W-1:0]  out_pkt,
    input  wire                  sched_err,    // sticky error set (any source): halt
    input  wire                  clear_error,  // mat_reset: abandon the list
    input  wire                  q_empty,      // scheduler input pipeline empty
    input  wire [3:0]            eng_busy,     // LD, ST, EX, VE
    output reg                   err_set,      // one-cycle pulse
    output reg  [3:0]            err_code,

    // status (CSR mirror)
    output reg  [31:0]           st_addr,      // address of the last decoded descriptor
    output reg  [31:0]           st_done,      // lists finished by END
    output reg  [31:0]           st_status,    // value of the last END
    output reg  [31:0]           st_err_idx,   // index (in its list) of a failing descriptor
    output reg  [31:0]           st_exec,      // descriptors decoded in the current list

    // AXI read (shares a DMA port through the arbiter in sa_unit)
    output reg  [31:0]           ar_addr,
    output reg                   ar_valid,
    input  wire                  ar_ready,
    input  wire [63:0]           r_data,
    input  wire [1:0]            r_resp,
    input  wire                  r_last,
    input  wire                  r_valid,
    output wire                  r_ready,

    // performance events: {AR stalled, waiting for descriptor words,
    //                      command waiting for the scheduler, descriptor decoded}
    output wire [3:0]            perf_ev
);
    `include "sa_defs.vh"
    localparam integer LOGD = $clog2(D);

    localparam [2:0] S_IDLE = 0, S_RUN = 1, S_FENCE = 2, S_DRAIN = 3, S_ERR = 4;
    reg  [2:0]  state;
    reg         resume;                 // DRAIN: continue at fetch_addr (JUMP) instead of stopping

    // ------------------------------------------------------------ fetch
    reg  [31:0] fetch_addr;
    reg  [2:0]  outstanding;            // bursts in flight (<= 4)
    reg  [63:0] fifo [0:31];
    reg  [5:0]  wp, rp;
    wire [5:0]  used = wp - rp;         // 6-bit difference: modulo-64 pointers
    wire        fetching = state == S_RUN || state == S_FENCE;
    // room for everything already requested plus one more burst
    wire        room = {1'b0, used} + {outstanding, 3'b000} + 7'd8 <= 7'd32;
    wire        issue = fetching && !ar_valid && outstanding < 3'd4 && room;
    wire        beat  = r_valid && r_ready;
    wire        keep  = state == S_RUN || state == S_FENCE;     // store beats (else drop)
    wire        rerr  = beat && keep && r_resp != 2'b00;
    assign r_ready = 1'b1;              // space was reserved when the burst was issued

    // --------------------------------------------------------- assemble
    reg  [63:0] w [0:7];
    reg  [2:0]  wi;                     // next word to fill
    reg         dfull;                  // w[0..7] hold a complete descriptor
    wire        pop = !dfull && used != 0 && keep;

    // ----------------------------------------------------------- decode
    wire [7:0]  op      = w[0][7:0];
    wire [31:0] hdr     = w[0][31:0];
    wire        reloc   = hdr[DF_RELOC];
    wire [1:0]  basesel = hdr[DF_BASESEL +: 2];
    wire        fbefore = hdr[DF_FENCE_BEFORE];
    wire        is_cmd  = op == DESC_LD || op == DESC_ST || op == DESC_EX || op == DESC_VE;
    wire        is_ctl  = op == DESC_FENCE || op == DESC_JUMP || op == DESC_END;
    wire        hdr_ok  = (is_cmd || is_ctl) && hdr[31:13] == 19'd0;
    wire [31:0] base    = bases[32*basesel +: 32];
    wire [31:0] ddr     = w[1][31:0] + (reloc ? base : 32'd0);
    wire [11:0] rep     = w[2][11:0];
    reg  [`SA_PKT_W-1:0] cmd_pkt;
    always @* begin
        case (op)
            DESC_LD: cmd_pkt = pkt_ld(ddr, w[2][31:0], w[2][47:32], w[2][63:48], w[3][31:0], w[3][33:32]);
            DESC_ST: cmd_pkt = pkt_st(ddr, w[2][31:0], w[2][47:32], w[2][63:48], w[3][31:0]);
            DESC_EX: cmd_pkt = pkt_ex(w[1][15:0], w[1][31:16], w[1][47:32], w[1][59:48], w[1][60],
                                      rep == 0 ? 12'd0 : rep - 12'd1, w[2][31:16], w[2][47:32], w[2][63:48]);
            default: cmd_pkt = pkt_ve(w[1][31:0], w[1][63:32], w[2][31:0], w[2][63:32] >> LOGD,
                                      w[3][7:0], w[3][13:8], w[3][31:16], w[3][47:32], w[3][52:48],
                                      w[4][31:0], w[4][63:32], w[5][31:0]);
        endcase
    end

    reg                   pkt_valid;
    reg [`SA_PKT_W-1:0]   pkt;
    assign out_valid = pkt_valid && !sched_err;
    assign out_pkt   = pkt;

    reg  [31:0] count_lim;              // 0 = no limit
    reg  [31:0] dec_addr;               // address of the descriptor in w[]
    reg  [3:0]  fence_mask;
    reg         fenced;                 // FENCE_BEFORE of the descriptor in w[] done
    wire [3:0]  fm = fence_mask == 0 ? 4'hF : fence_mask;
    wire        fence_done = q_empty && !pkt_valid && (eng_busy & fm) == 0;
    wire        decode = state == S_RUN && dfull && !pkt_valid;
    wire        limit_hit = count_lim != 0 && st_exec + 1 == count_lim;

    assign busy = state != S_IDLE || pkt_valid;

    task stop(input do_resume);
        begin
            state  <= S_DRAIN;
            resume <= do_resume;
        end
    endtask
    task fail(input [3:0] code);
        begin
            state    <= S_ERR;
            err_set  <= 1;
            err_code <= code;
        end
    endtask

    integer k;
    always @(posedge clk) begin
        err_set <= 0;
        if (!resetn) begin
            state <= S_IDLE; ar_valid <= 0; outstanding <= 0; wp <= 0; rp <= 0;
            wi <= 0; dfull <= 0; pkt_valid <= 0; fenced <= 0; resume <= 0;
            st_done <= 0; st_status <= 0; st_err_idx <= 0; st_exec <= 0; st_addr <= 0;
            err_code <= 0;
        end else begin
            // ---- AXI read address / data
            if (ar_valid && ar_ready) ar_valid <= 0;
            if (issue) begin
                ar_valid   <= 1;
                ar_addr    <= fetch_addr;
                fetch_addr <= fetch_addr + 32'd64;
            end
            outstanding <= outstanding + (ar_valid && ar_ready) - (beat && r_last);
            if (beat && keep) begin
                fifo[wp[4:0]] <= r_data;
                wp <= wp + 1;
            end
            if (pkt_valid && out_ready && out_valid) pkt_valid <= 0;

            // ---- assemble
            if (pop) begin
                w[wi] <= fifo[rp[4:0]];
                rp    <= rp + 1;
                wi    <= wi + 1;
                if (wi == 3'd7) dfull <= 1;
            end

            // ---- control
            case (state)
                S_IDLE:
                    if (submit) begin
                        st_exec   <= 0;
                        count_lim <= submit_count;
                        fence_mask <= 0;
                        fenced    <= 0;
                        if (submit_addr[5:0] != 0) begin
                            st_err_idx <= 0;
                            fail(XERR_SHAPE);
                        end else begin
                            state      <= S_RUN;
                            fetch_addr <= submit_addr;
                            dec_addr   <= submit_addr;
                        end
                    end
                S_RUN:
                    if (sched_err) begin
                        state <= S_ERR;                          // halted by another error
                    end else if (decode) begin
                        if (!hdr_ok) begin
                            st_err_idx <= st_exec;
                            fail(XERR_SHAPE);
                        end else if (is_cmd && fbefore && !fenced) begin
                            fence_mask <= 0;                     // wait for everything first
                            fenced     <= 1;
                            state      <= S_FENCE;
                        end else begin
                            st_addr  <= dec_addr;
                            st_exec  <= st_exec + 1;
                            dec_addr <= dec_addr + 32'd64;
                            dfull    <= 0;
                            wi       <= 0;
                            fenced   <= 0;
                            if (is_cmd) begin
                                pkt       <= cmd_pkt;
                                pkt_valid <= 1;
                                if (limit_hit) stop(0);
                            end else if (op == DESC_FENCE) begin
                                fence_mask <= w[1][3:0];
                                state      <= S_FENCE;
                            end else if (op == DESC_JUMP) begin
                                if (w[1][5:0] != 0) begin
                                    st_err_idx <= st_exec;
                                    fail(XERR_SHAPE);
                                end else begin
                                    fetch_addr <= w[1][31:0];
                                    dec_addr   <= w[1][31:0];
                                    stop(!limit_hit);
                                end
                            end else begin                       // END
                                st_status <= w[1][31:0];
                                st_done   <= st_done + 1;
                                stop(0);
                            end
                        end
                    end
                S_FENCE:
                    if (sched_err)
                        state <= S_ERR;
                    else if (fence_done) begin
                        if (count_lim != 0 && st_exec == count_lim) stop(0);    // a FENCE was the last one
                        else                                        state <= S_RUN;
                    end
                S_DRAIN:
                    // drop everything fetched after this point; go on once
                    // nothing is in flight any more
                    if (outstanding == 0 && !ar_valid && !(beat && r_last)) begin
                        rp    <= wp;
                        wi    <= 0;
                        dfull <= 0;
                        state <= resume ? S_RUN : S_IDLE;
                    end
                default:                                         // S_ERR
                    if (clear_error) begin
                        pkt_valid <= 0;
                        stop(0);
                    end
            endcase

            // (DRAIN neither stores nor pops; its exit empties the FIFO and w[])
            if (clear_error && state != S_ERR && state != S_IDLE && state != S_DRAIN) begin
                pkt_valid <= 0;                                  // mat_reset abandons the list
                stop(0);
            end
            if (rerr && state != S_ERR) begin                    // read response error (last word wins)
                st_err_idx <= st_exec;
                fail(XERR_RRESP);
            end
        end
    end

    assign perf_ev = {ar_valid && !ar_ready, state == S_RUN && !dfull && !pkt_valid,
                      pkt_valid && !out_ready, decode && hdr_ok};
endmodule
