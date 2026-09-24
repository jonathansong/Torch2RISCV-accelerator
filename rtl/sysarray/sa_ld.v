// LD engine: DDR -> local memory (docs/double_buffer_design.md §5).
//
// A command moves `rows` DDR rows of `row_bytes` (multiple of 8) spaced by
// `pitch`, starting at `ddr`, into memory `mem` at word `word`:
//   LINEAR     each DDR row starts at a new local word; bytes fill words in
//              order (lane = 64-bit piece of a word)
//   INTERLEAVE local word = word + chunk*rows + row, chunk = word-sized piece
//              of a row (A strips for the array; SPAD only)
// It is split into INCR bursts of <= 16 beats that never cross 4 KB, dealt
// round-robin to NPORTS AXI read ports (up to 4 outstanding per port). Each
// burst remembers its local start position, so returning beats are written
// by address in any port order. Commands run one at a time; `done` pulses
// after the last beat of the command is written, with `err` set if any beat
// came back with SLVERR/DECERR.
`timescale 1ns / 1ps
`include "sa_macros.vh"

module sa_ld #(
    parameter integer D      = 8,
    parameter integer NPORTS = 1
) (
    input  wire                 clk,
    input  wire                 resetn,

    input  wire                 cmd_valid,
    output wire                 cmd_ready,
    input  wire [31:0]          cmd_ddr,
    input  wire [3:0]           cmd_mem,
    input  wire [15:0]          cmd_word,
    input  wire [15:0]          cmd_rows,
    input  wire [15:0]          cmd_row_bytes,
    input  wire [31:0]          cmd_pitch,
    input  wire [1:0]           cmd_mode,
    output reg                  done,
    output reg                  err,
    output wire                 busy,

    // local write: one 64-bit lane of a word per cycle
    output wire                 lw_en,
    output reg  [3:0]           lw_mem,
    output wire [15:0]          lw_word,
    output wire [7:0]           lw_lane,
    output wire [63:0]          lw_data,

    output reg  [NPORTS*32-1:0] m_araddr,
    output reg  [NPORTS*8-1:0]  m_arlen,
    output reg  [NPORTS-1:0]    m_arvalid,
    input  wire [NPORTS-1:0]    m_arready,
    input  wire [NPORTS*64-1:0] m_rdata,
    input  wire [NPORTS*2-1:0]  m_rresp,
    input  wire [NPORTS-1:0]    m_rlast,
    input  wire [NPORTS-1:0]    m_rvalid,
    output wire [NPORTS-1:0]    m_rready
);
    `include "sa_defs.vh"
    localparam integer LOGD = $clog2(D);
    localparam integer PB   = NPORTS > 1 ? $clog2(NPORTS) : 1;
    localparam integer QD   = 8;                      // outstanding bursts per port

    // lanes per word (log2) of a memory
    function [3:0] lpw_log2(input [3:0] mem);
        lpw_log2 = mem == MEM_ACC  ? LOGD - 1 :       // 4D bytes = D/2 lanes
                   mem == MEM_DESC ? 0 : LOGD - 3;    // SPAD: D bytes = D/8 lanes
    endfunction

    // ---------------------------------------------------------- command
    reg         active;           // command accepted, not yet done
    reg         gen;              // still issuing bursts
    reg  [15:0] rows;
    reg  [31:0] row_bytes;       // may be flattened: rows x row_bytes
    reg  [31:0] pitch;
    reg  [3:0]  lpwl;
    reg  [15:0] step;             // local words between consecutive chunks of a row
    reg  [15:0] wpr;              // LINEAR: words per row
    reg  [15:0] r;                // current row
    reg  [31:0] off;              // byte offset in the row
    reg  [31:0] row_ddr;
    reg  [15:0] row_word;
    reg  [15:0] cur_word;
    reg  [7:0]  cur_lane;
    reg  [31:0] issued, completed;
    reg         rerr;

    assign cmd_ready = !active && resetn;

    wire [3:0] c_lpwl = lpw_log2(cmd_mem);
    wire       flat   = !cmd_mode[MODE_INTERLEAVE] && cmd_pitch == cmd_row_bytes &&
                        (cmd_row_bytes & ((16'd8 << c_lpwl) - 16'd1)) == 0;
    assign busy      = active;

    // next burst
    wire [31:0] addr      = row_ddr + off;
    wire [31:0] beats_row = (row_bytes - off) >> 3;
    wire [12:0] beats_4k  = (13'h1000 - {1'b0, addr[11:0]}) >> 3;
    wire [4:0]  beats     = beats_row < 16 && beats_row <= beats_4k ? beats_row[4:0] :
                            beats_4k < 16 ? beats_4k[4:0] : 5'd16;
    wire [8:0]  lane_sum  = cur_lane + beats;
    wire [15:0] words_adv = lane_sum >> lpwl;
    wire [7:0]  lane_mask = (8'd1 << lpwl) - 8'd1;

    // per-port burst queues: {word, lane, beats}
    reg  [15:0] q_word  [0:NPORTS-1][0:QD-1];
    reg  [7:0]  q_lane  [0:NPORTS-1][0:QD-1];
    reg  [3:0]  q_wp    [0:NPORTS-1];
    reg  [3:0]  q_rp    [0:NPORTS-1];
    wire [NPORTS-1:0] q_full, q_empty;

    reg  [PB-1:0] ar_rr;          // next port to issue on
    reg  [PB-1:0] r_rr;           // R side: round-robin start
    reg  [PB-1:0] r_sel;          // R side: port granted this cycle
    reg           r_any;

    genvar gp;
    generate
        for (gp = 0; gp < NPORTS; gp = gp + 1) begin : qst
            // 4-bit difference: modulo-16 pointers (a 32-bit compare breaks on wrap)
            wire [3:0] used = q_wp[gp] - q_rp[gp];
            assign q_full[gp]  = used == QD;
            assign q_empty[gp] = used == 0;
        end
    endgenerate

    wire issue_ok = gen && !m_arvalid[ar_rr] && !q_full[ar_rr];

    integer p;
    always @(posedge clk) begin
        done <= 0;
        if (!resetn) begin
            active    <= 0;
            gen       <= 0;
            m_arvalid <= 0;
            ar_rr     <= 0;
            for (p = 0; p < NPORTS; p = p + 1) q_wp[p] <= 0;
        end else begin
            for (p = 0; p < NPORTS; p = p + 1)
                if (m_arvalid[p] && m_arready[p]) m_arvalid[p] <= 0;

            if (cmd_valid && cmd_ready) begin
                active    <= 1;
                gen       <= 1;
                // contiguous rows that fill whole words map identically as one
                // long row: fewer, longer bursts (e.g. an 8x8 tile = 1 burst)
                if (flat) begin
                    rows      <= 1;
                    row_bytes <= cmd_rows * cmd_row_bytes;
                end else begin
                    rows      <= cmd_rows;
                    row_bytes <= cmd_row_bytes;
                end
                pitch     <= cmd_pitch;
                lw_mem    <= cmd_mem;
                lpwl      <= lpw_log2(cmd_mem);
                step      <= cmd_mode[MODE_INTERLEAVE] ? cmd_rows : 16'd1;
                wpr       <= (cmd_row_bytes + (16'd8 << lpw_log2(cmd_mem)) - 16'd1) >> (lpw_log2(cmd_mem) + 3);
                r         <= 0;
                off       <= 0;
                row_ddr   <= cmd_ddr;
                row_word  <= cmd_word;
                cur_word  <= cmd_word;
                cur_lane  <= 0;
                issued    <= 0;
                rerr      <= 0;
            end else if (issue_ok) begin
                m_arvalid[ar_rr]            <= 1;
                m_araddr[32*ar_rr +: 32]    <= addr;
                m_arlen[8*ar_rr +: 8]       <= beats - 1;
                q_word[ar_rr][q_wp[ar_rr][2:0]] <= cur_word;
                q_lane[ar_rr][q_wp[ar_rr][2:0]] <= cur_lane;
                q_wp[ar_rr]                 <= q_wp[ar_rr] + 1;
                issued                      <= issued + 1;
                ar_rr <= ar_rr == NPORTS - 1 ? 0 : ar_rr + 1;
                if (off + {beats, 3'b000} == row_bytes) begin      // next row
                    off     <= 0;
                    r       <= r + 1;
                    row_ddr <= row_ddr + pitch;
                    cur_lane <= 0;
                    if (step == 1) begin
                        row_word <= row_word + wpr;
                        cur_word <= row_word + wpr;
                    end else begin
                        row_word <= row_word + 1;
                        cur_word <= row_word + 1;
                    end
                    if (r + 1 == rows) gen <= 0;
                end else begin
                    off      <= off + {beats, 3'b000};
                    cur_word <= cur_word + words_adv * step;
                    cur_lane <= lane_sum[7:0] & lane_mask;
                end
            end else if (NPORTS > 1 && gen && (m_arvalid[ar_rr] || q_full[ar_rr])) begin
                ar_rr <= ar_rr == NPORTS - 1 ? 0 : ar_rr + 1;      // try the next port
            end

            if (active && !gen && completed == issued && !(cmd_valid && cmd_ready)) begin
                active <= 0;
                done   <= 1;
                err    <= rerr;
            end
            if (lw_en && m_rresp[2*r_sel +: 2] != 2'b00) rerr <= 1;
        end
    end

    // ------------------------------------------------------------ R side
    // one beat per cycle, round-robin over ports that have data
    integer s, cand;
    always @* begin
        r_any = 0;
        r_sel = 0;
        for (s = NPORTS - 1; s >= 0; s = s - 1) begin
            cand = (r_rr + s) % NPORTS;
            if (m_rvalid[cand] && !q_empty[cand]) begin
                r_any = 1;
                r_sel = cand;
            end
        end
    end

    reg  [4:0] beat_idx [0:NPORTS-1];     // beats already taken from the head burst
    wire [15:0] h_word = q_word[r_sel][q_rp[r_sel][2:0]];
    wire [7:0]  h_lane = q_lane[r_sel][q_rp[r_sel][2:0]];
    wire [8:0]  h_sum  = h_lane + beat_idx[r_sel];

    assign lw_en   = r_any;
    assign lw_word = h_word + (h_sum >> lpwl) * step;
    assign lw_lane = h_sum[7:0] & lane_mask;
    assign lw_data = m_rdata[64*r_sel +: 64];

    generate
        for (gp = 0; gp < NPORTS; gp = gp + 1) begin : rr
            assign m_rready[gp] = r_any && r_sel == gp;
        end
    endgenerate

    always @(posedge clk) begin
        if (!resetn) begin
            r_rr      <= 0;
            completed <= 0;
            for (p = 0; p < NPORTS; p = p + 1) begin
                q_rp[p]     <= 0;
                beat_idx[p] <= 0;
            end
        end else begin
            if (cmd_valid && cmd_ready) completed <= 0;
            if (r_any) begin
                r_rr <= r_sel == NPORTS - 1 ? 0 : r_sel + 1;
                if (m_rlast[r_sel]) begin
                    beat_idx[r_sel] <= 0;
                    q_rp[r_sel]     <= q_rp[r_sel] + 1;
                    completed       <= completed + 1;
                end else
                    beat_idx[r_sel] <= beat_idx[r_sel] + 1;
            end
        end
    end
endmodule
