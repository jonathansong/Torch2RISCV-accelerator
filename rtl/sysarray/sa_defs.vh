// Shared constants for the double-buffered accelerator (docs/double_buffer_design.md).
// Included inside module bodies; sa_macros.vh (guarded) makes it self-contained
// when Vivado analyses the header on its own.
`include "sa_macros.vh"

// Local memory ids (LADDR[31:28])
localparam [3:0] MEM_SPAD_A = 4'd1;
localparam [3:0] MEM_SPAD_B = 4'd2;
localparam [3:0] MEM_ACC    = 4'd3;
localparam [3:0] MEM_DESC   = 4'hF;   // internal: legacy descriptor registers (sequencer only)

// Engines
localparam integer ENG_LD = 0;
localparam integer ENG_ST = 1;
localparam integer ENG_EX = 2;
localparam integer ENG_VE = 3;        // M3 vector engine
localparam integer ENG_FETCH = 4;     // descriptor fetch unit (error reports only)

// Command types in the dispatch queue
localparam [1:0] CMD_LD = 2'd0;
localparam [1:0] CMD_ST = 2'd1;
localparam [1:0] CMD_EX = 2'd2;
localparam [1:0] CMD_VE = 2'd3;       // M3

// Transfer modes (LD_MODE)
localparam integer MODE_INTERLEAVE = 0;   // bit index
localparam integer MODE_ZERO_PAD   = 1;   // bit index (M4)

// mat_cfg keys
localparam [7:0] CFG_LD_ROWS      = 8'd0;
localparam [7:0] CFG_LD_ROW_BYTES = 8'd1;
localparam [7:0] CFG_LD_PITCH     = 8'd2;
localparam [7:0] CFG_LD_MODE      = 8'd3;
localparam [7:0] CFG_ST_ROWS      = 8'd4;
localparam [7:0] CFG_ST_ROW_BYTES = 8'd5;
localparam [7:0] CFG_ST_PITCH     = 8'd6;
localparam [7:0] CFG_EX_REPEAT    = 8'd7;    // M2: output tiles per mat_exec (1 .. 4095)
localparam [7:0] CFG_EX_B_STEP    = 8'd8;    //     SPAD_B words between tiles' B strips
localparam [7:0] CFG_EX_C_STEP    = 8'd9;    //     ACC words between tiles
localparam [7:0] CFG_EX_C_ROW     = 8'd10;   //     ACC words between rows of a tile
localparam [7:0] CFG_BASE0        = 8'd11;   // descriptor relocation bases BASE0..BASE15 (keys 11..26; L1: 16)
localparam [7:0] CFG_PARAM0       = 8'd27;   // L1: descriptor parameters PARAM0..PARAM7 (keys 27..34)

// vec_cfg keys (funct7 = 2, funct3 = 0), M3
localparam [7:0] VCFG_OP       = 8'd0;   // [2:0] op, [4] RELU, [5] REQUANT
localparam [7:0] VCFG_LEN      = 8'd1;   // elements (multiple of VL)
localparam [7:0] VCFG_DST      = 8'd2;   // destination LADDR
localparam [7:0] VCFG_TYPES    = 8'd3;   // [1:0] input type, [5:4] output type
localparam [7:0] VCFG_SRC2_MOD = 8'd4;   // src2 period in groups: 0 = none, 1 = broadcast
localparam [7:0] VCFG_SCALE    = 8'd5;   // REQUANT multiplier (int16)
localparam [7:0] VCFG_SHIFT    = 8'd6;   // REQUANT right shift (0 .. 31)
localparam [7:0] VCFG_ZP       = 8'd7;   // REQUANT zero point (int32)
localparam [7:0] VCFG_CLAMP_LO = 8'd8;   // result clamp (int32), then saturation to the type
localparam [7:0] VCFG_CLAMP_HI = 8'd9;

localparam [2:0] VOP_ADD = 3'd0, VOP_SUB = 3'd1, VOP_MUL = 3'd2, VOP_MAX = 3'd3,
                 VOP_MIN = 3'd4, VOP_COPY = 3'd5;
localparam [1:0] VT_I8 = 2'd0, VT_I16 = 2'd1, VT_I32 = 2'd2;   // I32 in ACC, I8/I16 in SPAD

// Sticky error codes (extended status bits 11:8)
localparam [3:0] XERR_SHAPE = 4'd1;   // bad address / shape / alignment
localparam [3:0] XERR_RANGE = 4'd2;   // local address out of range / bad memory id
localparam [3:0] XERR_RRESP = 4'd3;   // DMA read response SLVERR/DECERR
localparam [3:0] XERR_BRESP = 4'd4;   // DMA write response SLVERR/DECERR

// Dispatch packet: one command in the queues
//   [1:0]     type
//   [33:2]    ddr address           (LD/ST)
//   [65:34]   local address         (LD dst / ST src)
//   [81:66]   rows                  (LD/ST)
//   [97:82]   row_bytes             (LD/ST)
//   [129:98]  pitch                 (LD/ST)
//   [131:130] mode                  (LD)
//   [132]     internal              (issued by the legacy sequencer; allows MEM_DESC)
// VE (M3):
//   [33:2] src1 LADDR  [65:34] src2 LADDR  [97:66] dst LADDR  [113:98] groups (of VL elements)
//   [121:114] op byte  [127:122] types {out[3:2], in[1:0]}  [143:128] src2 period
//   [159:144] scale  [164:160] shift  [196:165] zp  [228:197] clamp lo  [260:229] clamp hi
// EX reuses the fields:
//   [17:2]    A word   [33:18] B word   [49:34] C word   [61:50] Kt   [62] accumulate
//   [74:63]   repeat - 1 (tiles)   [90:75] B step   [106:91] C step   [122:107] C row stride
//   (tile r: B strip at B + r*Bstep, C row i at C + r*Cstep + i*Crow; a zero C row
//    stride means 1, so a zero-filled extension is the M1 single-tile command)
localparam integer PKT_W = `SA_PKT_W;   // sa_macros.vh

// Performance counters (sa_perf.v; docs/perf_counters_and_desc_dma_plan.md §1.2).
// Read with mat_perf (funct7 = 1, funct3 = 5) or the CSR mirror at 0x40 + 4*i.
localparam integer PERF_NCNT = 32;
localparam integer PC_CYCLES = 0,  PC_CMD_LD = 1,  PC_CMD_ST = 2,  PC_CMD_EX = 3,  PC_CMD_VE = 4,
                   PC_PCPI_QFULL = 5, PC_PCPI_FENCE = 6,
                   PC_HAZ_LD = 7,  PC_HAZ_ST = 8,  PC_HAZ_EX = 9,  PC_HAZ_VE = 10,
                   PC_DISP_FULL = 11, PC_STARVE = 12, PC_ALL_IDLE = 13,
                   PC_EX_STEP = 14, PC_EX_USEFUL = 15, PC_EX_SWAPWAIT = 16, PC_EX_TILES = 17,
                   PC_LD_BUSY = 18, PC_LD_BEATS = 19, PC_LD_ARSTALL = 20,
                   PC_ST_BUSY = 21, PC_ST_BEATS = 22, PC_ST_WSTALL = 23,
                   PC_VE_ACTIVE = 24, PC_VE_RDBLOCK = 25, PC_VE_CREDIT = 26, PC_VE_GROUPS = 27;
                   // 28..31 reserved (descriptor DMA)

// Command packets (layout above). One definition for every producer: the
// PCPI decoder (sa_pcpi.v) and the descriptor fetch unit (sa_cmdfetch.v).
function [PKT_W-1:0] pkt_ld(input [31:0] ddr, input [31:0] laddr, input [15:0] rows,
                            input [15:0] rb, input [31:0] pitch, input [1:0] mode);
    pkt_ld = {1'b0, mode, pitch, rb, rows, laddr, ddr, CMD_LD};
endfunction
function [PKT_W-1:0] pkt_st(input [31:0] ddr, input [31:0] laddr, input [15:0] rows,
                            input [15:0] rb, input [31:0] pitch);
    pkt_st = {1'b0, 2'b00, pitch, rb, rows, laddr, ddr, CMD_ST};
endfunction
function [PKT_W-1:0] pkt_ex(input [15:0] a, input [15:0] b, input [15:0] c, input [11:0] kt,
                            input acc, input [11:0] rep_m1, input [15:0] bstep,
                            input [15:0] cstep, input [15:0] crow);
    pkt_ex = {10'd0, crow, cstep, bstep, rep_m1, acc, kt, c, b, a, CMD_EX};
endfunction
function [PKT_W-1:0] pkt_ve(input [31:0] src1, input [31:0] src2, input [31:0] dst,
                            input [15:0] groups, input [7:0] op, input [5:0] types,
                            input [15:0] period, input [15:0] scale, input [4:0] shift,
                            input [31:0] zp, input [31:0] lo, input [31:0] hi);
    pkt_ve = {hi, lo, zp, shift, scale, period, types, op, groups, dst, src2, src1, CMD_VE};
endfunction

// Descriptors (sa_cmdfetch.v; docs/double_buffer_design.md §8.6): 64 bytes =
// 8 little-endian 64-bit words w0..w7, 64-byte aligned. w0 = header.
localparam [7:0] DESC_LD = 8'h01, DESC_ST = 8'h02, DESC_EX = 8'h03, DESC_VE = 8'h04,
                 DESC_FENCE = 8'h10, DESC_JUMP = 8'h11, DESC_END = 8'h12,   // 0x00 is invalid
                 // L1 command extensions (docs/llm_inference_plan.md §5.3)
                 DESC_LOOP_END = 8'h13, DESC_SETREG = 8'h14, DESC_CALL = 8'h15, DESC_RET = 8'h16,
                 DESC_LDPARAM = 8'h17;
localparam integer DF_RELOC = 8;          // w0 flags: DDR address += BASE[BASESEL]
localparam integer DF_BASESEL = 9;        //           [10:9]
localparam integer DF_FENCE_BEFORE = 11;  //           wait for idle engines first
localparam integer DF_IRQ = 12;           //           END: reserved (ignored)
localparam integer DF_BASESEL_HI = 13;    //           [14:13] BASESEL bits 3:2 (L1: BASE0..15)
localparam integer DF_DYN0 = 16;          //           [23:16] dynamic slot 0, [31:24] slot 1 (L1):
                                          //           [3:0] field (0 = none) [6:4] PARAM [7] add
