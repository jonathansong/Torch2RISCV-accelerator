// Descriptor fetch unit (docs/perf_counters_and_desc_dma_plan.md part 2,
// docs/double_buffer_design.md §8.6; L1 command extensions:
// docs/llm_inference_plan.md §5.3).
//
// mat_submit(list, count) starts it: 64-byte descriptors are read from DDR
// (one 8-beat burst each, up to 4 in flight, into a 32 x 64-bit LUTRAM FIFO),
// assembled, patched, decoded and fed to the scheduler input in list order,
// exactly like the commands the PCPI decoder builds (the pkt_* functions of
// sa_defs.vh). Control descriptors are handled here:
//   FENCE    stop feeding until the scheduler queue is empty and the engines
//            in the mask (0 = all) are idle
//   JUMP     continue at an absolute (64-byte aligned) or relative address
//   END      record the status value, count a finished list, stop
//   L1:
//   SETREG   write up to three BASE / PARAM registers (replace or add)
//   LOOP_END jump back while iterations remain, adding strides to two
//            PARAMs each iteration (a 2-level stack keyed by the LOOP_END
//            address, so loops nest)
//   CALL/RET subroutines (return stack of 4)
//   LDPARAM  PARAM = mem32 * mul + add (the 64-byte block holding the word
//            is read on the same AXI port; prefetched descriptors are dropped
//            and fetched again after it)
// Registers: BASE0..15 (relocation; BASESEL = {w0[14:13], w0[10:9]}) and
// PARAM0..7, written by mat_cfg (keys 11..34) while no list runs, and by
// SETREG / LOOP_END / LDPARAM. Before decode, the two dynamic slots of w0
// (w0[23:16], w0[31:24]: [3:0] field, [6:4] PARAM, [7] add) replace or add
// to one field each (one cycle per used slot).
// A list also stops after `count` descriptors (count 0 = until END).
// Descriptors are prefetched speculatively; after a jump / END / the count /
// an error the unit drops the bursts still in flight (DRAIN) before it goes on.
//
// Errors (reported to the scheduler's sticky error with engine ENG_FETCH):
// a misaligned list / jump / LDPARAM address, an invalid header (opcode,
// reserved bit 15), an undefined dynamic field, a loop count of 0, a loop or
// call stack overflow / RET without CALL, a SETREG register > 23 ->
// XERR_SHAPE; a read response error -> XERR_RRESP. The unit halts until
// clear_error (mat_reset), then drains and goes idle. It also halts when the
// scheduler reports any other error.
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
    // mat_cfg keys 11..34 (only while !busy): 0..15 BASE0..15, 16..23 PARAM0..7
    input  wire                  reg_we,
    input  wire [4:0]            reg_idx,
    input  wire [31:0]           reg_val,

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

    // AXI read (shares a DMA port through the arbiter in sa_unit; 8-beat bursts)
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

    localparam [2:0] S_IDLE = 0, S_RUN = 1, S_FENCE = 2, S_DRAIN = 3, S_ERR = 4,
                     S_SETREG = 5, S_LDP = 6, S_LDPW = 7;
    reg  [2:0]  state;
    reg         resume;                 // DRAIN: continue at fetch_addr (jump) instead of stopping

    // ------------------------------------------------------- registers
    reg  [31:0] base  [0:15];
    reg  [31:0] param [0:7];

    // ------------------------------------------------------------ fetch
    reg  [31:0] fetch_addr;
    reg  [2:0]  outstanding;            // bursts in flight (<= 4)
    reg  [63:0] fifo [0:31];
    reg  [5:0]  wp, rp;
    wire [5:0]  used = wp - rp;         // 6-bit difference: modulo-64 pointers
    wire        fetching = state == S_RUN || state == S_FENCE || state == S_SETREG;
    // room for everything already requested plus one more burst
    wire        room = {1'b0, used} + {outstanding, 3'b000} + 7'd8 <= 7'd32;
    wire        issue = fetching && !ar_valid && outstanding < 3'd4 && room;
    wire        beat  = r_valid && r_ready;
    wire        keep  = fetching;       // store beats (else drop)
    wire        rerr  = beat && keep && r_resp != 2'b00;
    assign r_ready = 1'b1;              // space was reserved when the burst was issued

    // --------------------------------------------------------- assemble
    reg  [63:0] w [0:7];
    reg  [2:0]  wi;                     // next word to fill
    reg         dfull;                  // w[0..7] hold a complete descriptor
    wire        pop = !dfull && used != 0 && keep;

    // ------------------------------------------------ dynamic slots (patch)
    wire [7:0]  op      = w[0][7:0];
    wire [31:0] hdr     = w[0][31:0];
    reg  [1:0]  pdone;                  // slots of the descriptor in w[] applied
    wire [7:0]  sl0 = hdr[DF_DYN0 +: 8], sl1 = hdr[DF_DYN0 + 8 +: 8];
    wire        need0 = sl0[3:0] != 0 && !pdone[0];
    wire        need1 = sl1[3:0] != 0 && !pdone[1];
    wire        patching = dfull && (need0 || need1);
    wire [7:0]  slot = need0 ? sl0 : sl1;

    // field of (op, slot[3:0]) -> word, 16-bit lane, width (0: 12, 1: 16, 2: 32 bits)
    reg         f_ok;
    reg  [2:0]  f_word;
    reg  [1:0]  f_lane, f_wc;
    always @* begin
        f_ok = 1; f_word = 1; f_lane = 0; f_wc = 2;
        case (op)
            DESC_LD, DESC_ST:
                case (slot[3:0])
                    4'd1: begin f_word = 1; f_lane = 0; f_wc = 2; end   // DDR address
                    4'd2: begin f_word = 2; f_lane = 0; f_wc = 1; end   // local word
                    4'd3: begin f_word = 2; f_lane = 2; f_wc = 1; end   // rows
                    4'd4: begin f_word = 2; f_lane = 3; f_wc = 1; end   // row bytes
                    4'd5: begin f_word = 3; f_lane = 0; f_wc = 2; end   // pitch
                    default: f_ok = 0;
                endcase
            DESC_EX:
                case (slot[3:0])
                    4'd1: begin f_word = 1; f_lane = 0; f_wc = 1; end   // A
                    4'd2: begin f_word = 1; f_lane = 1; f_wc = 1; end   // B
                    4'd3: begin f_word = 1; f_lane = 2; f_wc = 1; end   // C
                    4'd4: begin f_word = 1; f_lane = 3; f_wc = 0; end   // Kt
                    4'd5: begin f_word = 2; f_lane = 0; f_wc = 0; end   // repeat
                    4'd6: begin f_word = 2; f_lane = 1; f_wc = 1; end   // B step
                    4'd7: begin f_word = 2; f_lane = 2; f_wc = 1; end   // C step
                    default: f_ok = 0;
                endcase
            DESC_VE:
                case (slot[3:0])
                    4'd1:  begin f_word = 1; f_lane = 0; f_wc = 1; end  // src1 word
                    4'd2:  begin f_word = 1; f_lane = 2; f_wc = 1; end  // src2 word
                    4'd3:  begin f_word = 2; f_lane = 0; f_wc = 1; end  // dst word
                    4'd4:  begin f_word = 2; f_lane = 2; f_wc = 2; end  // LEN
                    4'd5:  begin f_word = 7; f_lane = 1; f_wc = 1; end  // VALID (L2)
                    4'd6:  begin f_word = 7; f_lane = 0; f_wc = 1; end  // ROWLEN (L2)
                    4'd7:  begin f_word = 7; f_lane = 2; f_wc = 1; end  // P1 (L2)
                    4'd8:  begin f_word = 3; f_lane = 1; f_wc = 1; end  // P2 (period)
                    4'd9:  begin f_word = 6; f_lane = 0; f_wc = 2; end  // A (L2)
                    4'd10: begin f_word = 6; f_lane = 2; f_wc = 2; end  // B (L2)
                    4'd11: begin f_word = 5; f_lane = 2; f_wc = 2; end  // IMM (L2)
                    default: f_ok = 0;
                endcase
            DESC_JUMP, DESC_CALL, DESC_LDPARAM:
                if (slot[3:0] == 4'd1) begin f_word = 1; f_lane = 0; f_wc = 2; end
                else f_ok = 0;
            DESC_LOOP_END:
                if (slot[3:0] == 4'd1) begin f_word = 2; f_lane = 0; f_wc = 1; end
                else f_ok = 0;
            DESC_SETREG:
                if (slot[3:0] >= 4'd1 && slot[3:0] <= 4'd3) begin f_word = 3'd1 + slot[2:0]; f_lane = 0; f_wc = 2; end
                else f_ok = 0;
            default: f_ok = 0;
        endcase
    end
    wire [63:0] f_mask    = f_wc == 0 ? 64'hFFF : f_wc == 1 ? 64'hFFFF : 64'hFFFF_FFFF;
    wire [5:0]  f_sh      = {f_lane, 4'd0};
    wire [63:0] f_old     = (w[f_word] >> f_sh) & f_mask;
    wire [31:0] f_param   = param[slot[6:4]];
    wire [63:0] f_new     = (slot[7] ? f_old + {32'd0, f_param} : {32'd0, f_param}) & f_mask;
    wire [63:0] f_patched = (w[f_word] & ~(f_mask << f_sh)) | (f_new << f_sh);

    // ----------------------------------------------------------- decode
    wire        reloc   = hdr[DF_RELOC];
    wire [3:0]  basesel = {hdr[DF_BASESEL_HI +: 2], hdr[DF_BASESEL +: 2]};
    wire        fbefore = hdr[DF_FENCE_BEFORE];
    wire        is_cmd  = op == DESC_LD || op == DESC_ST || op == DESC_EX || op == DESC_VE;
    wire        is_ctl  = op == DESC_FENCE || op == DESC_JUMP || op == DESC_END ||
                          op == DESC_LOOP_END || op == DESC_SETREG || op == DESC_CALL ||
                          op == DESC_RET || op == DESC_LDPARAM;
    wire        hdr_ok  = (is_cmd || is_ctl) && !hdr[15];
    wire [31:0] base_v  = base[basesel];
    wire [31:0] ddr     = w[1][31:0] + (reloc ? base_v : 32'd0);
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
                                      w[4][31:0], w[4][63:32], w[5][31:0],
                                      w[3][63:53], w[5][63:32], w[6][31:0], w[6][63:32],
                                      w[7][15:0], w[7][31:16], w[7][47:32], w[7][63:48]);
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
    wire        decode = state == S_RUN && dfull && !pkt_valid && !patching;
    wire        limit_hit = count_lim != 0 && st_exec + 1 == count_lim;

    // jumps: relative targets count descriptors from this one
    wire        rel     = w[2][0];
    wire [31:0] rel_tgt = dec_addr + {w[1][25:0], 6'd0};
    wire [31:0] tgt     = rel ? rel_tgt : w[1][31:0];

    // loop stack (2 levels) and call stack (4)
    reg  [31:0] lp_addr [0:1];
    reg  [15:0] lp_rem  [0:1];
    reg  [1:0]  lp_n;
    wire        lp_top   = lp_n == 2'd2;                     // index of the innermost level
    wire        lp_match = lp_n != 0 && lp_addr[lp_top] == dec_addr;
    wire [15:0] lp_cnt   = w[2][15:0];
    wire [15:0] lp_cur   = lp_match ? lp_rem[lp_top] : lp_cnt;
    wire [2:0]  lp_k1    = w[2][18:16], lp_k2 = w[2][21:19];
    reg  [31:0] cs [0:3];
    reg  [2:0]  cs_n;
    wire [1:0]  cs_top   = cs_n[1:0] - 2'd1;

    // SETREG (latched at decode, one register per cycle)
    reg  [23:0] sr_sel;
    reg  [31:0] sr_val0, sr_val1, sr_val2;
    reg  [2:0]  sr_add;
    reg  [1:0]  sr_i;
    reg         sr_stop;
    wire [6:0]  sr_e   = sr_i == 0 ? sr_sel[6:0] : sr_i == 1 ? sr_sel[14:8] : sr_sel[22:16];
    wire [31:0] sr_val = sr_i == 0 ? sr_val0 : sr_i == 1 ? sr_val1 : sr_val2;
    wire [31:0] sr_cur = sr_e[4] ? param[sr_e[2:0]] : base[sr_e[3:0]];
    wire [31:0] sr_v   = sr_add[sr_i] ? sr_cur + sr_val : sr_val;

    // LDPARAM
    reg  [31:0] ldp_a, ldp_add, ldp_word;
    reg  [15:0] ldp_mul;
    reg  [2:0]  ldp_k, ldp_beat;
    reg         ldp_pend, ldp_stop, ldp_err;
    wire [31:0] ldp_data = ldp_beat == ldp_a[5:3] ? (ldp_a[2] ? r_data[63:32] : r_data[31:0]) : ldp_word;

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
            wi <= 0; dfull <= 0; pdone <= 0; pkt_valid <= 0; fenced <= 0; resume <= 0;
            st_done <= 0; st_status <= 0; st_err_idx <= 0; st_exec <= 0; st_addr <= 0;
            err_code <= 0; lp_n <= 0; cs_n <= 0; ldp_pend <= 0;
            for (k = 0; k < 16; k = k + 1) base[k] <= 0;
            for (k = 0; k < 8; k = k + 1) param[k] <= 0;
        end else begin
            if (reg_we) begin
                if (reg_idx[4]) param[reg_idx[2:0]] <= reg_val;
                else            base[reg_idx[3:0]]  <= reg_val;
            end

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
                        st_exec    <= 0;
                        count_lim  <= submit_count;
                        fence_mask <= 0;
                        fenced     <= 0;
                        lp_n       <= 0;
                        cs_n       <= 0;
                        ldp_pend   <= 0;
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
                    end else if (dfull && !pkt_valid && patching) begin
                        if (!hdr_ok || !f_ok) begin              // bad header / undefined dynamic field
                            st_err_idx <= st_exec;
                            fail(XERR_SHAPE);
                        end else begin
                            w[f_word] <= f_patched;
                            if (need0) pdone[0] <= 1;
                            else       pdone[1] <= 1;
                        end
                    end else if (decode) begin
                        if (!hdr_ok) begin
                            st_err_idx <= st_exec;
                            fail(XERR_SHAPE);
                        end else if ((is_cmd || op == DESC_LDPARAM) && fbefore && !fenced) begin
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
                            pdone    <= 0;
                            case (op)
                                DESC_LD, DESC_ST, DESC_EX, DESC_VE: begin
                                    pkt       <= cmd_pkt;
                                    pkt_valid <= 1;
                                    if (limit_hit) stop(0);
                                end
                                DESC_FENCE: begin
                                    fence_mask <= w[1][3:0];
                                    state      <= S_FENCE;
                                end
                                DESC_JUMP, DESC_CALL:
                                    if ((!rel && w[1][5:0] != 0) || (op == DESC_CALL && cs_n == 3'd4)) begin
                                        st_err_idx <= st_exec;
                                        fail(XERR_SHAPE);        // misaligned / call stack overflow
                                    end else begin
                                        if (op == DESC_CALL) begin
                                            cs[cs_n[1:0]] <= dec_addr + 32'd64;
                                            cs_n <= cs_n + 3'd1;
                                        end
                                        fetch_addr <= tgt;
                                        dec_addr   <= tgt;
                                        stop(!limit_hit);
                                    end
                                DESC_RET:
                                    if (cs_n == 0) begin
                                        st_err_idx <= st_exec;
                                        fail(XERR_SHAPE);        // RET without CALL
                                    end else begin
                                        fetch_addr <= cs[cs_top];
                                        dec_addr   <= cs[cs_top];
                                        cs_n       <= cs_n - 3'd1;
                                        stop(!limit_hit);
                                    end
                                DESC_SETREG: begin
                                    sr_sel  <= w[1][23:0];
                                    sr_val0 <= w[2][31:0];
                                    sr_val1 <= w[3][31:0];
                                    sr_val2 <= w[4][31:0];
                                    sr_add  <= w[5][2:0];
                                    sr_i    <= 0;
                                    sr_stop <= limit_hit;
                                    state   <= S_SETREG;
                                end
                                DESC_LOOP_END:
                                    if (lp_cnt == 0 || (!lp_match && lp_n == 2'd2)) begin
                                        st_err_idx <= st_exec;
                                        fail(XERR_SHAPE);        // count 0 / loop stack overflow
                                    end else begin
                                        if (lp_k1 == lp_k2)
                                            param[lp_k1] <= param[lp_k1] + w[3][31:0] + w[4][31:0];
                                        else begin
                                            param[lp_k1] <= param[lp_k1] + w[3][31:0];
                                            param[lp_k2] <= param[lp_k2] + w[4][31:0];
                                        end
                                        if (lp_cur > 16'd1) begin
                                            if (lp_match)
                                                lp_rem[lp_top] <= lp_cur - 16'd1;
                                            else begin
                                                lp_addr[lp_n[0]] <= dec_addr;
                                                lp_rem[lp_n[0]]  <= lp_cur - 16'd1;
                                                lp_n <= lp_n + 2'd1;
                                            end
                                            fetch_addr <= rel_tgt;
                                            dec_addr   <= rel_tgt;
                                            stop(!limit_hit);
                                        end else begin
                                            if (lp_match) lp_n <= lp_n - 2'd1;
                                            if (limit_hit) stop(0);
                                        end
                                    end
                                DESC_LDPARAM:
                                    if (ddr[1:0] != 0) begin
                                        st_err_idx <= st_exec;
                                        fail(XERR_SHAPE);
                                    end else begin
                                        ldp_a      <= ddr;       // w1 (+ BASE when relocated)
                                        ldp_k      <= w[2][2:0];
                                        ldp_mul    <= w[3][15:0];
                                        ldp_add    <= w[4][31:0];
                                        ldp_stop   <= limit_hit;
                                        ldp_pend   <= 1;
                                        fetch_addr <= dec_addr + 32'd64;   // refetch after the read
                                        stop(1);
                                    end
                                default: begin                   // END
                                    st_status <= w[1][31:0];
                                    st_done   <= st_done + 1;
                                    stop(0);
                                end
                            endcase
                        end
                    end
                S_SETREG:
                    if (sr_e[6] && sr_e[5:0] >= 6'd24) begin
                        st_err_idx <= st_exec - 1;
                        fail(XERR_SHAPE);
                    end else begin
                        if (sr_e[6]) begin
                            if (sr_e[4]) param[sr_e[2:0]] <= sr_v;
                            else         base[sr_e[3:0]]  <= sr_v;
                        end
                        sr_i <= sr_i + 2'd1;
                        if (sr_i == 2'd2) begin
                            if (sr_stop) stop(0);
                            else         state <= S_RUN;
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
                        pdone <= 0;
                        state <= ldp_pend ? S_LDP : resume ? S_RUN : S_IDLE;
                    end
                S_LDP: begin                                     // read the block holding the word
                    ar_valid <= 1;
                    ar_addr  <= {ldp_a[31:6], 6'd0};
                    ldp_beat <= 0;
                    ldp_err  <= 0;
                    state    <= S_LDPW;
                end
                S_LDPW:
                    if (beat) begin
                        ldp_beat <= ldp_beat + 3'd1;
                        ldp_word <= ldp_data;
                        if (r_resp != 2'b00) ldp_err <= 1;
                        if (r_last) begin
                            ldp_pend <= 0;
                            if (ldp_err || r_resp != 2'b00) begin
                                st_err_idx <= st_exec - 1;
                                fail(XERR_RRESP);
                            end else begin
                                param[ldp_k] <= ldp_data * {16'd0, ldp_mul} + ldp_add;
                                state <= ldp_stop ? S_IDLE : S_RUN;
                            end
                        end
                    end
                default:                                         // S_ERR
                    if (clear_error) begin
                        pkt_valid <= 0;
                        ldp_pend  <= 0;
                        stop(0);
                    end
            endcase

            // (DRAIN neither stores nor pops; its exit empties the FIFO and w[])
            if (clear_error && state != S_ERR && state != S_IDLE && state != S_DRAIN) begin
                pkt_valid <= 0;                                  // mat_reset abandons the list
                ldp_pend  <= 0;
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
