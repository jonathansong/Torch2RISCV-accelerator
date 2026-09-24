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
localparam integer ENG_VE = 3;        // reserved (M3)

// Command types in the dispatch queue
localparam [1:0] CMD_LD = 2'd0;
localparam [1:0] CMD_ST = 2'd1;
localparam [1:0] CMD_EX = 2'd2;

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
// EX reuses the fields:
//   [17:2]    A word   [33:18] B word   [49:34] C word   [61:50] Kt   [62] accumulate
//   [74:63]   repeat - 1 (tiles)   [90:75] B step   [106:91] C step   [122:107] C row stride
//   (tile r: B strip at B + r*Bstep, C row i at C + r*Cstep + i*Crow; a zero C row
//    stride means 1, so a zero-filled extension is the M1 single-tile command)
localparam integer PKT_W = `SA_PKT_W;   // sa_macros.vh
