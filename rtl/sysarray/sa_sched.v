// Command scheduler: input queue, validation, bank scoreboard, engine queues
// (docs/double_buffer_design.md §7).
//
// Commands arrive in program order (from the PCPI decoder or the legacy
// sequencer) and are dispatched in that order. At dispatch a command is
// validated and its bank read/write sets are computed (6 banks: SPAD_A,
// SPAD_B, ACC x 2). It waits while an engine *other* than its own has an
// in-flight command with a conflicting access (RAW, WAR, WAW); commands of
// the same engine run in order and are not checked against each other.
// An invalid command, or an engine reporting a bus error, sets a sticky
// error; dispatch stops until clear_error (mat_reset), which also flushes
// the input queue.
//
// Pipeline: input FIFO -> d1 (fields, span multiply) -> d2 (range/shape
// checks, bank masks) -> dispatch (hazard check, engine queue push). Each
// stage holds one command, so program order is preserved.
`timescale 1ns / 1ps
`include "sa_macros.vh"

module sa_sched #(
    parameter integer D          = 8,
    parameter integer SPAD_WORDS = 16384,
    parameter integer ACC_WORDS  = 8192
) (
    input  wire               clk,
    input  wire               resetn,

    input  wire               in_valid,
    output wire               in_ready,
    input  wire [`SA_PKT_W-1:0] in_pkt,

    // engine command queues (heads)
    output wire               ld_valid,
    input  wire               ld_ready,
    output wire [`SA_PKT_W-1:0] ld_pkt,
    output wire               st_valid,
    input  wire               st_ready,
    output wire [`SA_PKT_W-1:0] st_pkt,
    output wire               ex_valid,
    input  wire               ex_ready,
    output wire [`SA_PKT_W-1:0] ex_pkt,
    output wire               ve_valid,
    input  wire               ve_ready,
    output wire [`SA_PKT_W-1:0] ve_pkt,

    input  wire               ld_done,
    input  wire               ld_err,
    input  wire               st_done,
    input  wire               st_err,
    input  wire               ex_done,
    input  wire               ve_done,

    input  wire               clear_error,
    input  wire [3:0]         fence_mask,     // engines a fence waits for (0 = all)
    output wire               fence_ok,       // queue empty and masked engines idle
    output wire               idle,           // everything idle
    output wire [31:0]        ext_status,

    // performance events: [3:0] command dispatched, [7:4] head blocked by a
    // hazard (both by engine), [8] head blocked by a full engine queue,
    // [9] starved (no command at the head while an engine is busy), [10] all idle
    output wire [10:0]        perf_ev
);
    `include "sa_defs.vh"
    localparam integer LOGD = $clog2(D);
    localparam integer QIN  = 8;
    localparam integer QENG = 4;
    localparam integer QMSK = 8;

    // ---------------------------------------------------------- input FIFO
    reg  [PKT_W-1:0] inq [0:QIN-1];
    reg  [3:0]       in_wp, in_rp;
    wire [3:0]       in_used = in_wp - in_rp;
    wire             in_empty = in_used == 0;
    reg              err;
    reg  [3:0]       err_code, err_eng;

    assign in_ready = in_used != QIN && resetn;

    wire [PKT_W-1:0] fh = inq[in_rp[2:0]];

    function integer depth_of(input [3:0] mem);
        depth_of = mem == MEM_ACC ? ACC_WORDS : SPAD_WORDS;
    endfunction
    function [3:0] wb_log2(input [3:0] mem);         // bytes per word
        wb_log2 = mem == MEM_ACC ? LOGD + 2 : LOGD;
    endfunction
    // bank mask bits: SPAD_A 0/1, SPAD_B 2/3, ACC 4/5
    function [5:0] banks(input [3:0] mem, input [31:0] first, input [31:0] last);
        reg b0, b1;
        begin
            b0 = mem == MEM_ACC ? first >= ACC_WORDS / 2 : first >= SPAD_WORDS / 2;
            b1 = mem == MEM_ACC ? last  >= ACC_WORDS / 2 : last  >= SPAD_WORDS / 2;
            banks = 0;
            if (mem == MEM_SPAD_A) banks[1:0] = {b1, !b0};
            if (mem == MEM_SPAD_B) banks[3:2] = {b1, !b0};
            if (mem == MEM_ACC)    banks[5:4] = {b1, !b0};
        end
    endfunction

    // ------------------------------------------------ stage d1: fields + span
    reg              d1_v;
    reg [PKT_W-1:0]  d1;
    reg [31:0]       d1_span;          // LD/ST words;  EX: Kt*D
    reg [31:0]       d1_bext;          // EX: (repeat-1) * B step
    reg [31:0]       d1_cext;          // EX: (repeat-1) * C step + (D-1) * C row stride
    wire [3:0]  f_mem = fh[65:62];
    wire [15:0] f_rows = fh[81:66];
    wire [15:0] f_rb   = fh[97:82];
    wire [3:0]  f_wbl  = wb_log2(f_mem);
    wire [31:0] f_wpr  = (f_rb + (32'd1 << f_wbl) - 1) >> f_wbl;
    wire [15:0] f_crow = fh[122:107] == 0 ? 16'd1 : fh[122:107];
    wire [31:0] f_bext = fh[74:63] * fh[90:75];
    wire [31:0] f_cext = fh[74:63] * fh[106:91] + (D - 1) * f_crow;
    wire [31:0] f_span = fh[1:0] == CMD_EX ? fh[61:50] * D :
                         fh[130] ? (f_rb >> LOGD) * f_rows : f_wpr * f_rows;

    // ------------------------------------------------ stage d2: checks + banks
    wire [1:0]  h_type = d1[1:0];
    wire [31:0] h_ddr  = d1[33:2];
    wire [3:0]  h_mem  = d1[65:62];
    wire [15:0] h_word = d1[49:34];
    wire [15:0] h_rows = d1[81:66];
    wire [15:0] h_rb   = d1[97:82];
    wire [31:0] h_pitch = d1[129:98];
    wire [1:0]  h_mode = d1[131:130];
    wire        h_int  = d1[132];                    // internal (legacy sequencer)
    wire [15:0] h_a    = d1[17:2];
    wire [15:0] h_b    = d1[33:18];
    wire [15:0] h_c    = d1[49:34];
    wire [11:0] h_kt   = d1[61:50];
    wire        h_acc  = d1[62];

    wire        h_ilv     = h_mode[MODE_INTERLEAVE];
    wire [31:0] last_word = h_word + d1_span - 1;
    wire        mem_ok    = h_mem == MEM_SPAD_A || h_mem == MEM_SPAD_B || h_mem == MEM_ACC;
    wire        shape_ok  = h_ddr[2:0] == 0 && h_rb[2:0] == 0 && h_rb != 0 && h_rows != 0 &&
                            h_pitch[2:0] == 0 &&
                            (!h_ilv || (h_type == CMD_LD && h_mem != MEM_ACC && h_rb[LOGD-1:0] == 0));
    wire        range_ok  = mem_ok && last_word < depth_of(h_mem);
    wire [31:0] k_words   = d1_span;
    wire        ex_ok     = h_kt != 0;
    wire [31:0] b_last    = h_b + d1_bext + k_words - 1;
    wire [31:0] c_last    = h_c + d1_cext;
    wire        ex_range  = h_a + k_words <= SPAD_WORDS && b_last < SPAD_WORDS && c_last < ACC_WORDS;

    // VE command (M3)
    wire [3:0]  v_m1 = d1[33:30],  v_m2 = d1[65:62],  v_md = d1[97:94];
    wire [15:0] v_w1 = d1[17:2],   v_w2 = d1[49:34],  v_wd = d1[81:66];
    wire [15:0] v_groups = d1[113:98];
    wire [2:0]  v_op = d1[116:114];
    wire [1:0]  v_it = d1[123:122], v_ot = d1[125:124];
    wire [15:0] v_mod = d1[143:128];
    wire        v_unary = v_op == VOP_COPY;
    wire [31:0] v_wg_in  = v_it == VT_I16 ? 2 : 1;
    wire [31:0] v_wg_out = v_ot == VT_I16 ? 2 : 1;
    wire [31:0] v_g2 = v_mod == 0 ? v_groups : v_mod == 1 ? 1 : (v_mod < v_groups ? v_mod : v_groups);
    wire [31:0] v_last1 = v_w1 + v_groups * v_wg_in - 1;
    wire [31:0] v_last2 = v_w2 + v_g2 * v_wg_in - 1;
    wire [31:0] v_lastd = v_wd + v_groups * v_wg_out - 1;
    function mem_for(input [1:0] t, input [3:0] m);       // int32 in ACC, int8/int16 in SPAD
        mem_for = t == VT_I32 ? m == MEM_ACC : (m == MEM_SPAD_A || m == MEM_SPAD_B);
    endfunction
    wire        v_shape_ok = v_groups != 0 && v_it <= VT_I32 && v_ot <= VT_I32 && v_op <= VOP_COPY &&
                             !(v_op == VOP_MUL && v_it == VT_I32);
    wire        v_range_ok = mem_for(v_it, v_m1) && (v_unary || mem_for(v_it, v_m2)) && mem_for(v_ot, v_md) &&
                             v_last1 < depth_of(v_m1) && (v_unary || v_last2 < depth_of(v_m2)) &&
                             v_lastd < depth_of(v_md);

    reg         c_valid_cmd;
    reg  [3:0]  c_err_code;
    reg  [1:0]  c_eng;
    reg  [5:0]  c_r, c_w;
    always @* begin
        c_valid_cmd = 1;
        c_err_code  = 0;
        c_eng       = h_type == CMD_LD ? ENG_LD : h_type == CMD_ST ? ENG_ST :
                      h_type == CMD_EX ? ENG_EX : ENG_VE;
        c_r = 0;
        c_w = 0;
        case (h_type)
            CMD_LD:
                if (h_int && h_mem == MEM_DESC) begin
                    // legacy descriptor fetch into internal registers: no banks
                end else if (!shape_ok) begin
                    c_valid_cmd = 0; c_err_code = XERR_SHAPE;
                end else if (!range_ok) begin
                    c_valid_cmd = 0; c_err_code = XERR_RANGE;
                end else
                    c_w = banks(h_mem, h_word, last_word);
            CMD_ST:
                if (!shape_ok || h_ilv) begin
                    c_valid_cmd = 0; c_err_code = XERR_SHAPE;
                end else if (!range_ok) begin
                    c_valid_cmd = 0; c_err_code = XERR_RANGE;
                end else
                    c_r = banks(h_mem, h_word, last_word);
            CMD_EX:
                if (!ex_ok) begin
                    c_valid_cmd = 0; c_err_code = XERR_SHAPE;
                end else if (!ex_range) begin
                    c_valid_cmd = 0; c_err_code = XERR_RANGE;
                end else begin
                    c_r = banks(MEM_SPAD_A, h_a, h_a + k_words - 1) |
                          banks(MEM_SPAD_B, h_b, b_last) |
                          (h_acc ? banks(MEM_ACC, h_c, c_last) : 6'd0);
                    c_w = banks(MEM_ACC, h_c, c_last);
                end
            CMD_VE:
                if (!v_shape_ok) begin
                    c_valid_cmd = 0; c_err_code = XERR_SHAPE;
                end else if (!v_range_ok) begin
                    c_valid_cmd = 0; c_err_code = XERR_RANGE;
                end else begin
                    c_r = banks(v_m1, v_w1, v_last1) | (v_unary ? 6'd0 : banks(v_m2, v_w2, v_last2));
                    c_w = banks(v_md, v_wd, v_lastd);
                end
            default: begin
                c_valid_cmd = 0; c_err_code = XERR_SHAPE;
            end
        endcase
    end

    reg              d2_v;
    reg [PKT_W-1:0]  h;                // the command at dispatch
    reg              h_valid_cmd;
    reg [3:0]        h_err_code;
    reg [1:0]        h_eng;
    reg [5:0]        h_r, h_w;

    // ------------------------------------------- in-flight masks per engine
    reg  [5:0] mr [0:3][0:QMSK-1];
    reg  [5:0] mw [0:3][0:QMSK-1];
    reg  [3:0] m_wp [0:3];
    reg  [3:0] m_rp [0:3];
    reg  [5:0] eng_r [0:3];
    reg  [5:0] eng_w [0:3];
    wire [3:0] m_empty, m_full;
    genvar ge;
    generate
        for (ge = 0; ge < 4; ge = ge + 1) begin : msk
            wire [3:0] used = m_wp[ge] - m_rp[ge];
            assign m_empty[ge] = used == 0;
            assign m_full[ge]  = used == QMSK;
            integer x;
            always @* begin
                eng_r[ge] = 0;
                eng_w[ge] = 0;
                for (x = 0; x < QMSK; x = x + 1)
                    if (((x - m_rp[ge]) & 7) < used) begin      // slot x holds a live entry
                        eng_r[ge] = eng_r[ge] | mr[ge][x];
                        eng_w[ge] = eng_w[ge] | mw[ge][x];
                    end
            end
        end
    endgenerate

    // hazard against the other engines (RAW, WAR, WAW); EX and VE also share
    // the memory ports of SPAD side B / ACC side A, so any common bank
    // between them conflicts, even two readers (M3)
    reg hazard;
    integer o;
    always @* begin
        hazard = 0;
        for (o = 0; o < 4; o = o + 1)
            if (o != h_eng) begin
                hazard = hazard | |(h_r & eng_w[o]) | |(h_w & eng_r[o]) | |(h_w & eng_w[o]);
                if ((h_eng == ENG_EX && o == ENG_VE) || (h_eng == ENG_VE && o == ENG_EX))
                    hazard = hazard | |((h_r | h_w) & (eng_r[o] | eng_w[o]));
            end
    end

    // ------------------------------------------------------ engine queues
    reg  [PKT_W-1:0] eq [0:3][0:QENG-1];
    reg  [2:0]       e_wp [0:3];
    reg  [2:0]       e_rp [0:3];
    wire [3:0]       e_empty, e_full;
    generate
        for (ge = 0; ge < 4; ge = ge + 1) begin : eqs
            wire [2:0] used = e_wp[ge] - e_rp[ge];
            assign e_empty[ge] = used == 0;
            assign e_full[ge]  = used == QENG;
        end
    endgenerate

    assign ld_valid = !e_empty[ENG_LD];
    assign st_valid = !e_empty[ENG_ST];
    assign ex_valid = !e_empty[ENG_EX];
    assign ld_pkt   = eq[ENG_LD][e_rp[ENG_LD][1:0]];
    assign st_pkt   = eq[ENG_ST][e_rp[ENG_ST][1:0]];
    assign ex_pkt   = eq[ENG_EX][e_rp[ENG_EX][1:0]];
    assign ve_valid = !e_empty[ENG_VE];
    assign ve_pkt   = eq[ENG_VE][e_rp[ENG_VE][1:0]];
    wire [3:0] e_pop = {ve_valid && ve_ready, ex_valid && ex_ready, st_valid && st_ready, ld_valid && ld_ready};
    wire [3:0] e_done = {ve_done, ex_done, st_done, ld_done};

    wire dispatch = d2_v && !err && h_valid_cmd && !hazard &&
                    !e_full[h_eng] && !m_full[h_eng];
    wire reject   = d2_v && !err && !h_valid_cmd;
    wire d2_free  = !d2_v || dispatch || reject;
    wire d1_move  = d1_v && d2_free && !err;
    wire d1_free  = !d1_v || d1_move;
    wire fifo_pop = !in_empty && d1_free && !err;

    integer e;
    always @(posedge clk) begin
        if (!resetn) begin
            in_wp <= 0; in_rp <= 0;
            d1_v <= 0; d2_v <= 0;
            err <= 0; err_code <= 0; err_eng <= 0;
            for (e = 0; e < 4; e = e + 1) begin
                m_wp[e] <= 0; m_rp[e] <= 0; e_wp[e] <= 0; e_rp[e] <= 0;
            end
        end else begin
            if (in_valid && in_ready) begin
                inq[in_wp[2:0]] <= in_pkt;
                in_wp <= in_wp + 1;
            end
            // pipeline: FIFO -> d1 -> d2 -> dispatch
            if (fifo_pop) begin
                d1      <= fh;
                d1_span <= f_span;
                d1_bext <= f_bext;
                d1_cext <= f_cext;
                in_rp   <= in_rp + 1;
            end
            if (d1_free) d1_v <= fifo_pop;
            if (d1_move) begin
                h           <= d1;
                h_valid_cmd <= c_valid_cmd;
                h_err_code  <= c_err_code;
                h_eng       <= c_eng;
                h_r         <= c_r;
                h_w         <= c_w;
            end
            if (d2_free) d2_v <= d1_move;
            if (dispatch) begin
                eq[h_eng][e_wp[h_eng][1:0]] <= h;
                e_wp[h_eng]                  <= e_wp[h_eng] + 1;
                mr[h_eng][m_wp[h_eng][2:0]]  <= h_r;
                mw[h_eng][m_wp[h_eng][2:0]]  <= h_w;
                m_wp[h_eng]                  <= m_wp[h_eng] + 1;
            end
            if (reject) begin
                err <= 1; err_code <= h_err_code; err_eng <= h_eng;
            end
            for (e = 0; e < 4; e = e + 1) begin
                if (e_pop[e])  e_rp[e] <= e_rp[e] + 1;
                if (e_done[e]) m_rp[e] <= m_rp[e] + 1;
            end
            if (ld_done && ld_err && !err) begin err <= 1; err_code <= XERR_RRESP; err_eng <= ENG_LD; end
            if (st_done && st_err && !err) begin err <= 1; err_code <= XERR_BRESP; err_eng <= ENG_ST; end
            if (clear_error) begin                           // flush everything not dispatched
                err   <= 0;
                in_rp <= in_wp + (in_valid && in_ready);     // (keep a push of this cycle)
                d1_v  <= 0;
                d2_v  <= 0;
            end
        end
    end

    // -------------------------------------------------------------- status
    wire [3:0] eng_busy = ~m_empty;
    wire [3:0] fm       = fence_mask == 0 ? 4'b1111 : fence_mask;
    wire       pipe_empty = in_empty && !d1_v && !d2_v;
    wire [4:0] queued     = in_used + d1_v + d2_v;
    assign idle     = pipe_empty && m_empty == 4'b1111;
    assign fence_ok = pipe_empty && (eng_busy & fm) == 0;
    assign ext_status = {8'd0, 3'd0, queued, err_eng, err_code, eng_busy, 2'b00, err, idle};

    wire       head_ok = d2_v && !err && h_valid_cmd;
    wire [3:0] h_onehot = 4'b0001 << h_eng;
    assign perf_ev = {pipe_empty && eng_busy == 4'd0,
                      !d2_v && eng_busy != 4'd0,
                      head_ok && !hazard && (e_full[h_eng] || m_full[h_eng]),
                      {4{head_ok && hazard}} & h_onehot,
                      {4{dispatch}} & h_onehot};
endmodule
