// sa HAL driver: the `board` transport (docs/iree_compiler_plan.md §5.1, §5.4).
//
// rt_fw (firmware/rt) serves a submission ring in DDR (llm_inference_plan.md
// §5.1): 64-byte entries {type | flags | seq << 32, list address, count,
// BASE0..3, parameter block}, 32-byte completion records {seq, status,
// cycles, descriptors, END value}, doorbell = mailbox RING_TAIL, progress =
// mailbox RING_HEAD. This transport owns the ring; the memory window, the
// mailbox and a running rt_fw come from the caller:
//
//   sa_board_attach(mem, mem_phys, mem_size, mailbox, ring_entries, d)
//     mem / mailbox: pointers already mapped in this process (a PYNQ launcher
//     loads this library with ctypes and passes its buffers' addresses; the
//     memory window must be non-cacheable, since this code does no cache
//     maintenance). Layout of the window: ring at 0 (ring_entries * 64 bytes),
//     completion records at 0x2000 (ring_entries * 32 bytes), the arena from
//     64 KB on. The launcher points rt_fw's mailbox RING_BASE / RING_SIZE /
//     CPL_BASE there and starts it before the first dispatch.
//
// Without sa_board_attach, the transport maps both itself through /dev/mem
// (a static test program started by a PYNQ launcher, compiler/runtime/test/
// board_launcher.py, which allocated the window, set up the ring and started
// rt_fw):
//   SA_BOARD_MEM=<physical address>:<bytes>   the window (page aligned)
//   SA_BOARD_MEM_DEV=<device file>            optional: map the window from this
//       file (offset 0) instead of /dev/mem at its physical address. On the
//       KV260 (arm64, CONFIG_STRICT_DEVMEM) /dev/mem refuses RAM even to root:
//       the window comes from the u-dma-buf module (/dev/udmabuf0, opened with
//       O_SYNC: a non-cached mapping), its physical address from sysfs.
//   SA_BOARD_MBOX=<physical address>          the mailbox (BRAM + 0x1F00)
//   SA_BOARD_RING=<entries>  SA_BOARD_D=<array size>
// O_SYNC makes the DDR mapping non-cacheable (ARM: write-combining normal
// memory for RAM, strongly ordered for the BRAM), so no cache maintenance.
//
// Addresses (docs/kv260_upgrade_plan.md §2.4): physical addresses are parsed
// as 64-bit values and checked over the whole range before anything narrows
// them. The window must end at or below 2 GB: descriptors and ring entries
// carry 32-bit DDR addresses, and the PicoRV32 reaches DDR only below its CSR
// window at 0x8000_0000 (on the KV260 the high 2 GB of DDR start at
// 0x8_0000_0000 and must not fold back). The mailbox is only mapped by this
// process, so it may be anywhere (a 64-bit PL window on the ZynqMP).
// SA_BOARD_DEVMEM=<file> replaces /dev/mem (the ring emulator on the host).
//
// Completion is polled (RING_HEAD); the notify interrupt is not used here.
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "sa_context.h"

enum {
  MBOX_RING_BASE = 0xB0, MBOX_RING_SIZE = 0xB4, MBOX_RING_TAIL = 0xB8, MBOX_RING_HEAD = 0xBC,
  MBOX_CPL_BASE = 0xC0, MBOX_FW_STATE = 0xC4,
};
#define RT_READY 0x52554E00u
#define RT_RUN_LIST 0x01u
#define RT_F_PERF (1u << 9)                       // rt_fw copies the event counters into its perf area
#define MBOX_PERF_COUNT 0x8Cu
#define SA_PERF_AREA_BELOW_MBOX 0x100u            // perf area BRAM + 0x1E00, mailbox BRAM + 0x1F00
#define SA_BOARD_CPL_OFFSET 0x2000u
#define SA_BOARD_DDR_LIMIT 0x80000000ull           // the window ends at or below 2 GB (see the top)
#define SA_BOARD_HEAP_OFFSET 0x10000u
// The ring protocol's layout (rt_fw, firmware/include/mailbox.h) is the same
// bytes on armv7 and aarch64: 64-byte entries of 64-bit words and 32-byte
// completion records of 32-bit words, written by offset (no structs, pointers
// or longs cross to the firmware).
_Static_assert(SA_BOARD_CPL_OFFSET >= 128u * 64u && SA_BOARD_CPL_OFFSET + 128u * 32u <= SA_BOARD_HEAP_OFFSET,
               "a ring of up to 128 entries, then its completion records, before the arena");

typedef struct sa_board_t {
  volatile uint32_t* mbox;
  volatile uint32_t* perf_area;  // SA_PROFILE_PERF: just below the mailbox, else NULL
  uint32_t ring_entries;
  uint32_t tail;           // entries submitted
} sa_board_t;

static struct {
  void* mem;
  uint32_t mem_phys, mem_size, ring_entries, d;
  void* mailbox;
} sa_board_attached;

void sa_board_attach(void* mem, uint32_t mem_phys, uint32_t mem_size, void* mailbox, uint32_t ring_entries,
                     uint32_t d) {
  sa_board_attached.mem = mem;
  sa_board_attached.mem_phys = mem_phys;
  sa_board_attached.mem_size = mem_size;
  sa_board_attached.mailbox = mailbox;
  sa_board_attached.ring_entries = ring_entries;
  sa_board_attached.d = d;
}

static uint32_t mbox_rd(sa_board_t* b, uint32_t off) { return b->mbox[off / 4]; }
static void mbox_wr(sa_board_t* b, uint32_t off, uint32_t v) { b->mbox[off / 4] = v; }

static double sa_now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static iree_status_t sa_board_run(sa_transport_t* t, uint32_t list_phys, sa_completion_t* out) {
  sa_board_t* b = (sa_board_t*)t->impl;
  uint32_t seq = b->tail, slot = seq % b->ring_entries;
  volatile uint64_t* e = (volatile uint64_t*)(t->mem + 64u * slot);
  volatile uint32_t* c = (volatile uint32_t*)(t->mem + SA_BOARD_CPL_OFFSET + 32u * slot);
  e[0] = (uint64_t)(RT_RUN_LIST | (b->perf_area ? RT_F_PERF : 0u)) | (uint64_t)seq << 32;
  e[1] = list_phys;
  for (int i = 2; i < 8; ++i) e[i] = 0;          // count 0 (to END), BASE0..3 0, no parameter block
  __sync_synchronize();
  b->tail = seq + 1;
  mbox_wr(b, MBOX_RING_TAIL, b->tail);           // doorbell
  double t0 = sa_now();
  while ((int32_t)(mbox_rd(b, MBOX_RING_HEAD) - b->tail) < 0) {
    if (sa_now() - t0 > 10.0) {
      return iree_make_status(IREE_STATUS_DEADLINE_EXCEEDED,
                              "sa board: entry %u not completed (RING_HEAD %u, FW_STATE %#x)", seq,
                              mbox_rd(b, MBOX_RING_HEAD), mbox_rd(b, MBOX_FW_STATE));
    }
  }
  __sync_synchronize();
  if (c[0] != seq) {
    return iree_make_status(IREE_STATUS_DATA_LOSS, "sa board: completion record %u has seq %u", seq, c[0]);
  }
  out->status = c[1];
  out->cycles = c[2];
  out->descriptors = c[3];
  out->end = c[4];
  out->perf_n = 0;
  if (b->perf_area) {                            // written by rt_fw before the completion record
    uint32_t n = mbox_rd(b, MBOX_PERF_COUNT);
    out->perf_n = n > 32 ? 32 : n;
    for (uint32_t k = 0; k < out->perf_n; ++k) out->perf[k] = b->perf_area[k];
  }
  return iree_ok_status();
}

static void sa_board_close(sa_transport_t* t) { (void)t; }

// Maps [phys, phys + size) through /dev/mem (O_SYNC); returns NULL on failure
// (also when the page base does not fit off_t: 32-bit off_t on armv7).
static void* sa_devmem_map(uint64_t phys, uint64_t size) {
  uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE);
  uint64_t base = phys & ~(page - 1);
  if (sizeof(off_t) < 8 && base > (uint64_t)INT32_MAX) return NULL;
  if (size + (phys - base) > (uint64_t)SIZE_MAX) return NULL;
  const char* devmem = getenv("SA_BOARD_DEVMEM");  // compiler/sim/sa_board_emu.py
  int fd = open(devmem ? devmem : "/dev/mem", O_RDWR | O_SYNC);
  if (fd < 0) return NULL;
  void* p = mmap(NULL, (size_t)(size + (phys - base)), PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)base);
  close(fd);
  return p == MAP_FAILED ? NULL : (uint8_t*)p + (phys - base);
}

// Maps [0, size) of a device file (u-dma-buf) with O_SYNC; NULL on failure.
static void* sa_file_map(const char* path, uint64_t size) {
  if (size > (uint64_t)SIZE_MAX) return NULL;
  int fd = open(path, O_RDWR | O_SYNC);
  if (fd < 0) return NULL;
  void* p = mmap(NULL, (size_t)size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  return p == MAP_FAILED ? NULL : p;
}

// A whole unsigned number (decimal, 0x hex, 0 octal) into 64 bits; *end: the
// first character after it (NULL: nothing may follow). Returns 0 on failure.
static int sa_parse_u64(const char* s, uint64_t* out, const char** end) {
  if (!s || !*s || *s == '-') return 0;
  char* e = NULL;
  errno = 0;
  unsigned long long v = strtoull(s, &e, 0);
  if (errno || e == s || (!end && *e)) return 0;
  if (end) *end = e;
  *out = (uint64_t)v;
  return 1;
}

// The device window [phys, phys + size): a valid range below SA_BOARD_DDR_LIMIT.
static iree_status_t sa_board_check_window(uint64_t phys, uint64_t size) {
  if (size == 0 || phys > SA_BOARD_DDR_LIMIT || size > SA_BOARD_DDR_LIMIT - phys) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "sa board: memory window [%#llx, %#llx) must be non-empty and lie below %#llx (32-bit device "
                            "addresses; the PicoRV32 reaches DDR below 2 GB)",
                            (unsigned long long)phys, (unsigned long long)(phys + size),
                            (unsigned long long)SA_BOARD_DDR_LIMIT);
  }
  return iree_ok_status();
}

// SA_BOARD_* (see the top of this file) -> sa_board_attach.
static iree_status_t sa_board_attach_from_env(void) {
  const char* mem = getenv("SA_BOARD_MEM");
  const char* mbox = getenv("SA_BOARD_MBOX");
  if (!mem || !mbox) return iree_ok_status();
  uint64_t phys = 0, size = 0, mbox_phys = 0, ring = 0, d = 0;
  const char* colon = NULL;
  if (!sa_parse_u64(mem, &phys, &colon) || *colon != ':' || !sa_parse_u64(colon + 1, &size, NULL) ||
      !sa_parse_u64(mbox, &mbox_phys, NULL) || !sa_parse_u64(getenv("SA_BOARD_RING"), &ring, NULL) ||
      !sa_parse_u64(getenv("SA_BOARD_D"), &d, NULL)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SA_BOARD_MEM=<phys>:<bytes>, SA_BOARD_MBOX=<phys>, SA_BOARD_RING=<entries> and "
                            "SA_BOARD_D=<array size> are needed (unsigned numbers)");
  }
  // the whole range checked before the 32-bit device addresses are formed
  IREE_RETURN_IF_ERROR(sa_board_check_window(phys, size));
  if (mbox_phys < SA_PERF_AREA_BELOW_MBOX || ring == 0 || ring > 128 || (ring & (ring - 1)) || (d != 8 && d != 16)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "sa board: SA_BOARD_MBOX %#llx, SA_BOARD_RING %llu (a power of 2 up to 128), "
                            "SA_BOARD_D %llu (8 or 16)",
                            (unsigned long long)mbox_phys, (unsigned long long)ring, (unsigned long long)d);
  }
  const char* memdev = getenv("SA_BOARD_MEM_DEV");
  void* m = memdev && *memdev ? sa_file_map(memdev, size) : sa_devmem_map(phys, size);
  // the mailbox and the perf area below it (SA_PROFILE_PERF)
  void* b = sa_devmem_map(mbox_phys - SA_PERF_AREA_BELOW_MBOX, 0x100 + SA_PERF_AREA_BELOW_MBOX);
  if (b) b = (uint8_t*)b + SA_PERF_AREA_BELOW_MBOX;
  if (!m || !b) {
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "sa board: cannot map the window %#llx (%s) or the mailbox %#llx (/dev/mem); run as "
                            "root (arm64 with CONFIG_STRICT_DEVMEM: the window needs SA_BOARD_MEM_DEV, u-dma-buf)",
                            (unsigned long long)phys, memdev && *memdev ? memdev : "/dev/mem",
                            (unsigned long long)mbox_phys);
  }
  sa_board_attach(m, (uint32_t)phys, (uint32_t)size, b, (uint32_t)ring, (uint32_t)d);
  return iree_ok_status();
}

iree_status_t sa_transport_board_open(iree_allocator_t host_allocator, sa_transport_t** out) {
  if (!sa_board_attached.mem) {
    IREE_RETURN_IF_ERROR(sa_board_attach_from_env());
  }
  if (!sa_board_attached.mem || !sa_board_attached.mailbox) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "sa board: no board attached (SA_BOARD_* or sa_board_attach)");
  }
  IREE_RETURN_IF_ERROR(sa_board_check_window(sa_board_attached.mem_phys, sa_board_attached.mem_size));
  sa_transport_t* t = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*t) + sizeof(sa_board_t), (void**)&t));
  memset(t, 0, sizeof(*t) + sizeof(sa_board_t));
  sa_board_t* b = (sa_board_t*)(t + 1);
  b->mbox = (volatile uint32_t*)sa_board_attached.mailbox;
  b->ring_entries = sa_board_attached.ring_entries;
  if (getenv("SA_PROFILE_PERF"))                 // (the BRAM is mapped from the perf area on)
    b->perf_area = (volatile uint32_t*)((uint8_t*)sa_board_attached.mailbox - SA_PERF_AREA_BELOW_MBOX);
  if (mbox_rd(b, MBOX_FW_STATE) != RT_READY) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION, "sa board: rt_fw not ready (FW_STATE %#x)",
                            mbox_rd(b, MBOX_FW_STATE));
  }
  if (mbox_rd(b, MBOX_RING_BASE) != sa_board_attached.mem_phys ||
      mbox_rd(b, MBOX_CPL_BASE) != sa_board_attached.mem_phys + SA_BOARD_CPL_OFFSET ||
      mbox_rd(b, MBOX_RING_SIZE) != b->ring_entries) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "sa board: the ring is not at the start of the memory window");
  }
  b->tail = mbox_rd(b, MBOX_RING_HEAD);          // continue after whatever ran before
  t->name = "board";
  t->mem = (uint8_t*)sa_board_attached.mem;
  t->mem_phys = sa_board_attached.mem_phys;
  t->mem_size = sa_board_attached.mem_size;
  t->heap_offset = SA_BOARD_HEAP_OFFSET;
  t->d = sa_board_attached.d;
  t->run = sa_board_run;
  t->close = sa_board_close;
  t->impl = b;
  *out = t;
  return iree_ok_status();
}
