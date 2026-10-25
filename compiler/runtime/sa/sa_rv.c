// sa HAL driver: dispatches through the generated RISC-V code (plan A,
// docs/riscv_backend_design.md §9.4): SA_RV=embedded takes each executable's
// own code (iree-compile --iree-sa-riscv, S4); SA_RV=<dir> holds a model's code image
// (compiler/tests/rv_image.py: rvcode.bin, rvcode.map, rvcode_host.so for the
// sim). The image is loaded once into the device window; a dispatch whose
// export has a function there runs as that function (the board: rt_fw's
// RT_CALL entry; the sim: the host build of the same code, its commands run on
// the functional simulator) instead of a descriptor list. SA_RV_ONLY=<a,b>:
// only exports whose name contains one of these (testing).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sa_context.h"

typedef struct sa_rv_entry_t {
  char* symbol;
  uint32_t offset;
} sa_rv_entry_t;

static struct {
  int state;                 // 0: not loaded, 1: off, 2: on
  uint32_t code_phys;
  uint8_t* code;
  sa_rv_entry_t* entries;
  uint32_t count;
  uint32_t* scratch;         // the argument block (64 words) and a 2-descriptor list (the sim)
  uint32_t scratch_phys;
  char only[512];
} sa_rv;

static int sa_rv_cmp(const void* a, const void* b) {
  return strcmp(((const sa_rv_entry_t*)a)->symbol, ((const sa_rv_entry_t*)b)->symbol);
}

static iree_status_t sa_rv_load(sa_context_t* c) {
  const char* dir = getenv("SA_RV");
  sa_rv.state = 1;
  if (!dir || !*dir) return iree_ok_status();
  if (strcmp(dir, "embedded") == 0) {                // S4: each executable's own code (sa_loader.c)
    if (!c->transport->run_call || strcmp(c->transport->name, "board") != 0) {   // (the sim runs host builds only)
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "sa rv: the %s transport cannot call embedded RISC-V code (the board, rt_fw >= 4)",
                              c->transport->name);
    }
    const char* only = getenv("SA_RV_ONLY");
    if (only) snprintf(sa_rv.only, sizeof(sa_rv.only), "%s", only);
    // the argument block and the transport's scratch (the image's are after its code)
    void* raw = NULL;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(sa_context_data_allocator(c), 64 + 64 * 4 + 256, &raw));
    sa_rv.scratch = (uint32_t*)(((uintptr_t)raw + 63) & ~(uintptr_t)63);
    IREE_RETURN_IF_ERROR(sa_context_phys(c, sa_rv.scratch, &sa_rv.scratch_phys));
    sa_rv.state = 3;
    fprintf(stderr, "sa rv: the executables' embedded RISC-V code\n");
    return iree_ok_status();
  }
  if (!c->transport->run_call) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "sa rv: the %s transport cannot call RISC-V code (rt_fw >= 4)",
                            c->transport->name);
  }
  const char* only = getenv("SA_RV_ONLY");
  if (only) snprintf(sa_rv.only, sizeof(sa_rv.only), "%s", only);
  char path[1024];
  snprintf(path, sizeof(path), "%s/rvcode.bin", dir);
  FILE* f = fopen(path, "rb");
  if (!f) return iree_make_status(IREE_STATUS_NOT_FOUND, "sa rv: cannot open %s", path);
  fseek(f, 0, SEEK_END);
  long size = ftell(f);
  fseek(f, 0, SEEK_SET);
  iree_allocator_t a = sa_context_data_allocator(c);
  void* raw = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(a, (iree_host_size_t)size + 64 + 64 * 4 + 256, &raw));
  uint8_t* code = (uint8_t*)(((uintptr_t)raw + 63) & ~(uintptr_t)63);
  size_t got = fread(code, 1, (size_t)size, f);
  fclose(f);
  if (got != (size_t)size) return iree_make_status(IREE_STATUS_DATA_LOSS, "sa rv: short read of %s", path);
  sa_rv.code = code;
  IREE_RETURN_IF_ERROR(sa_context_phys(c, code, &sa_rv.code_phys));
  sa_rv.scratch = (uint32_t*)(code + (((uint32_t)size + 63) & ~63u));
  IREE_RETURN_IF_ERROR(sa_context_phys(c, sa_rv.scratch, &sa_rv.scratch_phys));
  snprintf(path, sizeof(path), "%s/rvcode.map", dir);
  f = fopen(path, "r");
  if (!f) return iree_make_status(IREE_STATUS_NOT_FOUND, "sa rv: cannot open %s", path);
  uint32_t cap = 0;
  char sym[512];
  unsigned off;
  while (fscanf(f, "%x %511s", &off, sym) == 2) {
    if (sa_rv.count == cap) {
      cap = cap ? 2 * cap : 256;
      sa_rv.entries = (sa_rv_entry_t*)realloc(sa_rv.entries, cap * sizeof(*sa_rv.entries));
    }
    sa_rv.entries[sa_rv.count].symbol = strdup(sym);
    sa_rv.entries[sa_rv.count++].offset = off;
  }
  fclose(f);
  qsort(sa_rv.entries, sa_rv.count, sizeof(*sa_rv.entries), sa_rv_cmp);
  sa_rv.state = 2;
  fprintf(stderr, "sa rv: %u functions, %ld bytes of RISC-V code at %#x (%s)\n", sa_rv.count, size,
          sa_rv.code_phys, dir);
  return iree_ok_status();
}

// The argument block and the transport's scratch, once (SA_RV=embedded allocates
// them at load; an export with no descriptor form needs them without SA_RV).
static iree_status_t sa_rv_scratch_alloc(sa_context_t* c) {
  if (sa_rv.scratch) return iree_ok_status();
  void* raw = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(sa_context_data_allocator(c), 64 + 64 * 4 + 256, &raw));
  sa_rv.scratch = (uint32_t*)(((uintptr_t)raw + 63) & ~(uintptr_t)63);
  return sa_context_phys(c, sa_rv.scratch, &sa_rv.scratch_phys);
}

// The function of an export (the symbol: the export's name with every
// character but letters and digits as '_', SahwToRiscv.cpp riscvSymbol).
// |required|: the export has no descriptor form (the MoE Top-2): its RISC-V
// code whatever SA_RV and SA_RV_ONLY say, the executable's embedded code on the
// board, else the SA_RV directory's (the sim runs its host build); 0 if neither.
// SA_RV_ONLY=rv_only: only such exports run RISC-V code.
iree_status_t sa_rv_lookup(sa_context_t* c, iree_string_view_t name, char* symbol, size_t cap,
                           uint32_t embedded_phys, bool required, uint32_t* out_phys) {
  *out_phys = 0;
  if (sa_rv.state == 0) {
    IREE_RETURN_IF_ERROR(sa_rv_load(c));
  }
  if (name.size + 1 > cap) return iree_ok_status();
  for (size_t i = 0; i < name.size; ++i) {
    char ch = name.data[i];
    symbol[i] = ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9')) ? ch : '_';
  }
  symbol[name.size] = 0;
  if (required) {
    if (embedded_phys && c->transport->run_call && strcmp(c->transport->name, "board") == 0) {
      IREE_RETURN_IF_ERROR(sa_rv_scratch_alloc(c));
      *out_phys = embedded_phys;
    } else if (sa_rv.state == 2) {
      sa_rv_entry_t key = {symbol, 0};
      sa_rv_entry_t* e = (sa_rv_entry_t*)bsearch(&key, sa_rv.entries, sa_rv.count, sizeof(*sa_rv.entries), sa_rv_cmp);
      if (e) *out_phys = sa_rv.code_phys + e->offset;
    }
    return iree_ok_status();
  }
  if (sa_rv.state != 2 && sa_rv.state != 3) return iree_ok_status();
  if (strcmp(sa_rv.only, "rv_only") == 0) return iree_ok_status();
  if (sa_rv.only[0]) {
    bool hit = false;
    char list[512];
    snprintf(list, sizeof(list), "%s", sa_rv.only);
    for (char* t = strtok(list, ","); t && !hit; t = strtok(NULL, ",")) hit = strstr(symbol, t) != NULL;
    if (!hit) return iree_ok_status();
  }
  if (sa_rv.state == 3) {
    *out_phys = embedded_phys;
    return iree_ok_status();
  }
  sa_rv_entry_t key = {symbol, 0};
  sa_rv_entry_t* e = (sa_rv_entry_t*)bsearch(&key, sa_rv.entries, sa_rv.count, sizeof(*sa_rv.entries), sa_rv_cmp);
  if (e) *out_phys = sa_rv.code_phys + e->offset;
  return iree_ok_status();
}

// The symbol of the function at |phys| (the sim's schedule emulation calls it by name), or NULL.
const char* sa_rv_symbol_at(uint32_t phys) {
  for (uint32_t i = 0; i < sa_rv.count; ++i)
    if (sa_rv.code_phys + sa_rv.entries[i].offset == phys) return sa_rv.entries[i].symbol;
  return NULL;
}

static uint32_t sa_rv_host_phys;
static sa_rv_host_fn_t sa_rv_host_fn;
void sa_rv_set_host_kernel(uint32_t phys, sa_rv_host_fn_t fn) {
  sa_rv_host_phys = phys;
  sa_rv_host_fn = fn;
}
sa_rv_host_fn_t sa_rv_host_kernel_at(uint32_t phys) { return phys && phys == sa_rv_host_phys ? sa_rv_host_fn : NULL; }

// The transport's scratch for a call: 128 bytes after the argument block (sa_transport_sim.c's one-command lists).
uint32_t sa_rv_scratch_phys(void) { return sa_rv.scratch_phys; }

iree_status_t sa_rv_call(sa_context_t* c, iree_string_view_t name, const char* symbol, uint32_t code_phys,
                         uint32_t binding_count, const uint32_t* binding_phys, uint32_t constant_count,
                         const uint32_t* constants) {
  if (binding_count > 16 || constant_count > 48) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "sa rv: '%.*s': %u bindings, %u constants", (int)name.size,
                            name.data, binding_count, constant_count);
  }
  IREE_RETURN_IF_ERROR(sa_context_batch_flush(c));   // (the dispatches before it, in order)
  iree_slim_mutex_lock(&c->mutex);
  uint32_t block[64] = {0};
  for (uint32_t i = 0; i < 16; ++i) block[i] = i < binding_count ? binding_phys[i] : 0;
  for (uint32_t i = 0; i < constant_count; ++i) block[16 + i] = constants[i];
  if (c->record) {                                   // (plan B's recording: the call as an item)
    sa_record_add(c->record, code_phys, block, 64);
    if (c->record->dry) {
      c->dispatches++;
      c->rv_calls++;
      iree_slim_mutex_unlock(&c->mutex);
      return iree_ok_status();
    }
  }
  volatile uint32_t* args = sa_rv.scratch;
  for (uint32_t i = 0; i < 64; ++i) args[i] = block[i];
  __sync_synchronize();
  sa_completion_t done = {0};
  uint64_t t0 = iree_time_now();
  iree_status_t status = c->transport->run_call(c->transport, code_phys, sa_rv.scratch_phys, symbol, &done);
  uint64_t host_ns = (uint64_t)(iree_time_now() - t0);
  c->dispatches++;
  c->rv_calls++;
  iree_slim_mutex_unlock(&c->mutex);
  if (iree_status_is_ok(status) && done.status) {
    status = iree_make_status(IREE_STATUS_INTERNAL, "sa rv: '%.*s' (RISC-V code) failed on the %s device, status %#x",
                              (int)name.size, name.data, c->transport->name, done.status);
  }
  if (iree_status_is_ok(status)) sa_context_profile_record(c, name, done.cycles, host_ns, &done);
  return status;
}
