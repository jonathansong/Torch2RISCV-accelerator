// sa HAL driver: the sa-desc-v1 executable loader (docs/iree_compiler_plan.md §4, §5.2).
//
// Loading copies the exports' descriptor templates into device-visible memory
// (the context's arena), so each export has a physical entry address. A
// dispatch is one descriptor list built by sa_context_dispatch: SETREG of the
// registers, CALL the template, END. The registers: by default BASE i =
// physical address of binding i, PARAM j = push constant j; an export of
// version 2 may carry a register setup table instead (BASE r = binding b +
// f(push constant), PARAM r = f(push constant), f(c) = ((c * mul) >> shift) + add),
// which is how IREE's dynamic binding offsets reach the template. The template covers the whole dispatch, so only workgroup
// (0, 0, 0) issues it; the other workgroups of the grid are no-ops.
//
// File layout: compiler/runtime/tools/sadesc.py.
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "iree/hal/api.h"
#include "iree/hal/local/executable_loader.h"
#include "iree/hal/local/local_executable.h"
#include "sa_context.h"
#include "sa_loader.h"

#define SA_FORMAT "sa-desc-v1"
static const char sa_magic[8] = {'S', 'A', 'D', 'E', 'S', 'C', '1', 0};

typedef struct sa_file_header_t {
  char magic[8];
  uint32_t version, d, caps, export_count, export_offset;
  uint32_t templates_offset, templates_bytes, strings_offset, strings_bytes;
  uint32_t ext_offset;      // version 3: the export extensions
  uint32_t reserved[4];
} sa_file_header_t;
static_assert(sizeof(sa_file_header_t) == 64, "sa-desc-v1 header");

typedef struct sa_file_export_t {
  uint32_t name_offset, name_length, template_offset, descriptor_count;
  uint16_t binding_count, constant_count;
  uint32_t setup_count, est_cycles, setup_offset;
} sa_file_export_t;
static_assert(sizeof(sa_file_export_t) == 32, "sa-desc export entry");

// version 3, per export (compiler/runtime/tools/sadesc.py): the prefix
// (descriptors, its RET included; 0: none), the bindings it reads, the
// bindings written / read, the head (descriptors, RET included; 0: none),
// flags (bit 0: touches SPAD_B), the register the prefix was generated for
typedef struct sa_file_ext_t {
  uint32_t prefix_count, prefix_reads, writes, reads, head_count, flags, prefix_reg, reserved;
} sa_file_ext_t;
static_assert(sizeof(sa_file_ext_t) == 32, "sa-desc export extension");

enum { SA_SETUP_BASE = 0, SA_SETUP_PARAM = 1 };
typedef struct sa_setup_t {
  uint8_t kind;       // [3:0] SA_SETUP_*, [7:4] right shift
  uint8_t reg, binding, reserved0;
  int16_t constant;   // push constant ordinal, -1: none
  uint16_t reserved1;
  int32_t mul, add;
} sa_setup_t;
static_assert(sizeof(sa_setup_t) == 16, "sa-desc setup entry");

//===----------------------------------------------------------------------===//
// sa_executable_t
//===----------------------------------------------------------------------===//

typedef struct sa_export_t {
  iree_string_view_t name;  // points into the executable's copy of the strings
  uint32_t prefix_phys;      // 0: no prefix
  uint32_t head_phys;        // 0: no head
  uint32_t entry_phys;       // the body (after the head)
  uint32_t prefix_reads, writes, reads;   // binding masks
  uint32_t prefix_reg;
  bool spad_b;
  uint32_t descriptor_count;
  uint32_t est_cycles;
  uint32_t setup_count;     // 0: the default setup
  const sa_setup_t* setup;  // in the executable's host allocation
} sa_export_t;

typedef struct sa_executable_t {
  iree_hal_local_executable_t base;
  sa_context_t* context;
  void* templates;          // in the arena (data allocator)
  iree_host_size_t export_count;
  sa_export_t* exports;
  iree_hal_executable_dispatch_attrs_v0_t* attrs;
  char* strings;
} sa_executable_t;

static const iree_hal_local_executable_vtable_t sa_executable_vtable;

static sa_executable_t* sa_executable_cast(iree_hal_executable_t* e) { return (sa_executable_t*)e; }

static void sa_executable_destroy(iree_hal_executable_t* base) {
  sa_executable_t* e = sa_executable_cast(base);
  iree_allocator_t host = e->base.host_allocator;
  if (e->templates) iree_allocator_free_aligned(sa_context_data_allocator(e->context), e->templates);
  iree_hal_local_executable_deinitialize(&e->base);
  iree_allocator_free(host, e);
}

// The register values of one dispatch (the setup table or the default).
static iree_status_t sa_export_regs(sa_executable_t* e, const sa_export_t* x, uint32_t binding_count,
                                    void* const* binding_ptrs, uint32_t constant_count, const uint32_t* constants,
                                    uint32_t* regs, uint32_t* vals, uint32_t* out_n) {
  uint32_t phys[16], n = 0;
  if (binding_count > 16) return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "sa: more than 16 bindings");
  for (uint32_t i = 0; i < binding_count; ++i) {
    IREE_RETURN_IF_ERROR(sa_context_phys(e->context, binding_ptrs[i], &phys[i]), "binding %u of '%.*s'", i,
                         (int)x->name.size, x->name.data);
  }
  if (x->setup_count == 0) {
    if (constant_count > 6) return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "sa: more than 6 constants");
    for (uint32_t i = 0; i < binding_count; ++i) regs[n] = i, vals[n++] = phys[i];
    for (uint32_t i = 0; i < constant_count; ++i) regs[n] = SA_REG_PARAM + i, vals[n++] = constants[i];
  } else {
    for (uint32_t i = 0; i < x->setup_count; ++i) {
      const sa_setup_t* s = &x->setup[i];
      int64_t v = s->add;
      if (s->constant >= 0) v += ((int64_t)constants[s->constant] * s->mul) >> (s->kind >> 4);
      if ((s->kind & 15) == SA_SETUP_BASE) {
        regs[n] = s->reg;
        vals[n++] = phys[s->binding] + (uint32_t)v;
      } else {
        regs[n] = SA_REG_PARAM + s->reg;
        vals[n++] = (uint32_t)v;
      }
    }
  }
  *out_n = n;
  return iree_ok_status();
}

// One dispatch submitted and waited for (inline execution; profiling).
// The dispatch of export x with these registers and bindings (ranges: 48 entries of scratch).
static void sa_export_dispatch(const sa_export_t* x, uint32_t n, const uint32_t* regs, const uint32_t* vals,
                               uint32_t binding_count, void* const* binding_ptrs,
                               const iree_device_size_t* binding_lengths, sa_range_t* ranges, sa_dispatch_t* out) {
  memset(out, 0, sizeof(*out));
  out->prefix_phys = x->prefix_phys;
  out->head_phys = x->head_phys;
  out->body_phys = x->entry_phys;
  out->count = n;
  out->regs = regs;
  out->values = vals;
  out->spad_b = x->spad_b;
  out->name = x->name;
  for (uint32_t i = 0; i < n; ++i)
    if (regs[i] == x->prefix_reg) out->prefix_base = vals[i];
  sa_range_t* rd = ranges;
  sa_range_t* wr = ranges + 16;
  sa_range_t* pr = ranges + 32;
  for (uint32_t i = 0; i < binding_count && i < 16; ++i) {
    uintptr_t b = (uintptr_t)binding_ptrs[i];
    sa_range_t r = {b, b + (uintptr_t)binding_lengths[i]};
    if (x->reads >> i & 1) rd[out->read_count++] = r;
    if (x->writes >> i & 1) wr[out->write_count++] = r;
    if (x->prefix_reads >> i & 1) pr[out->prefix_read_count++] = r;
  }
  out->reads = rd;
  out->writes = wr;
  out->prefix_reads = pr;
}

static iree_status_t sa_export_run_now(sa_executable_t* e, const sa_export_t* x, uint32_t n, const uint32_t* regs,
                                       const uint32_t* vals) {
  sa_dispatch_t dispatch;
  sa_range_t ranges[48];
  sa_export_dispatch(x, n, regs, vals, 0, NULL, NULL, ranges, &dispatch);
  sa_completion_t done = {0};
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  IREE_RETURN_IF_ERROR(sa_context_dispatch(e->context, &dispatch, &done));
  clock_gettime(CLOCK_MONOTONIC, &t1);
  sa_context_profile_record(e->context, x->name, done.cycles,
                            (uint64_t)(t1.tv_sec - t0.tv_sec) * 1000000000ull + (uint64_t)(t1.tv_nsec - t0.tv_nsec));
  if (done.status != 0) {
    // extended status (rtl/sysarray/sa_sched.v): [11:8] code, [15:12] engine
    static const char* engines[] = {"LD", "ST", "EX", "VE", "FETCH"};
    static const char* codes[] = {"none", "shape", "range", "read response", "write response"};
    uint32_t eng = (done.status >> 12) & 0xF, code = (done.status >> 8) & 0xF;
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "sa: dispatch '%.*s' failed on the %s device: %s %s error (status %#x, %u "
                            "descriptors decoded)",
                            (int)x->name.size, x->name.data, e->context->transport->name,
                            eng < 5 ? engines[eng] : "?", code < 5 ? codes[code] : "?", done.status,
                            done.descriptors);
  }
  if (done.end != 0x5A) {
    return iree_make_status(IREE_STATUS_INTERNAL, "sa: dispatch '%.*s' ended with %#x (expected 0x5a)",
                            (int)x->name.size, x->name.data, done.end);
  }
  return iree_ok_status();
}

static iree_status_t sa_executable_issue_call(iree_hal_local_executable_t* base, iree_host_size_t ordinal,
                                              const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
                                              const iree_hal_executable_workgroup_state_v0_t* workgroup_state,
                                              uint32_t worker_id) {
  sa_executable_t* e = (sa_executable_t*)base;
  if (IREE_UNLIKELY(ordinal >= e->export_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "sa: export ordinal %u out of range",
                            (unsigned)ordinal);
  }
  // one descriptor list per dispatch
  if (workgroup_state->workgroup_id_x | workgroup_state->workgroup_id_y | workgroup_state->workgroup_id_z) {
    return iree_ok_status();
  }
  const sa_export_t* x = &e->exports[ordinal];
  uint32_t regs[24], vals[24], n = 0;
  IREE_RETURN_IF_ERROR(sa_export_regs(e, x, dispatch_state->binding_count, dispatch_state->binding_ptrs,
                                      dispatch_state->constant_count, dispatch_state->constants, regs, vals, &n));
  return sa_export_run_now(e, x, n, regs, vals);
}

bool sa_executable_isa(iree_hal_executable_t* executable);

iree_status_t sa_executable_append(iree_hal_executable_t* executable, uint32_t ordinal, uint32_t binding_count,
                                   void* const* binding_ptrs, const iree_device_size_t* binding_lengths,
                                   uint32_t constant_count, const uint32_t* constants) {
  if (!sa_executable_isa(executable)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "sa: the executable is not an sa-desc executable");
  }
  sa_executable_t* e = (sa_executable_t*)executable;
  if (ordinal >= e->export_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "sa: export ordinal %u out of range", ordinal);
  }
  const sa_export_t* x = &e->exports[ordinal];
  if (binding_count != e->attrs[ordinal].binding_count || constant_count != e->attrs[ordinal].constant_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "sa: '%.*s' takes %u bindings / %u constants",
                            (int)x->name.size, x->name.data, e->attrs[ordinal].binding_count,
                            e->attrs[ordinal].constant_count);
  }
  uint32_t regs[24], vals[24], n = 0;
  IREE_RETURN_IF_ERROR(
      sa_export_regs(e, x, binding_count, binding_ptrs, constant_count, constants, regs, vals, &n));
  // SA_NO_BATCH, SA_PROFILE (except SA_PROFILE=batch): one list per dispatch
  static int immediate = -1;
  if (immediate < 0) {
    const char* profile = getenv("SA_PROFILE");
    immediate = getenv("SA_NO_BATCH") || (profile && strcmp(profile, "batch") != 0);
  }
  if (immediate) return sa_export_run_now(e, x, n, regs, vals);
  sa_dispatch_t dispatch;
  sa_range_t ranges[48];
  sa_export_dispatch(x, n, regs, vals, binding_count, binding_ptrs, binding_lengths, ranges, &dispatch);
  return sa_context_batch_add(e->context, &dispatch);
}

static iree_host_size_t sa_executable_export_count(iree_hal_executable_t* base) {
  return sa_executable_cast(base)->export_count;
}

static iree_status_t sa_executable_export_info(iree_hal_executable_t* base,
                                               iree_hal_executable_export_ordinal_t ordinal,
                                               iree_hal_executable_export_info_t* out_info) {
  sa_executable_t* e = sa_executable_cast(base);
  memset(out_info, 0, sizeof(*out_info));
  if (ordinal >= e->export_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "sa: export ordinal %u out of range", ordinal);
  }
  out_info->name = e->exports[ordinal].name;
  out_info->constant_count = e->attrs[ordinal].constant_count;
  out_info->binding_count = e->attrs[ordinal].binding_count;
  out_info->parameter_count = 0;
  out_info->workgroup_size[0] = out_info->workgroup_size[1] = out_info->workgroup_size[2] = 1;
  return iree_ok_status();
}

static iree_status_t sa_executable_export_parameters(iree_hal_executable_t* base,
                                                     iree_hal_executable_export_ordinal_t ordinal,
                                                     iree_host_size_t capacity,
                                                     iree_hal_executable_export_parameter_t* out_parameters) {
  // no parameter reflection (parameter_count 0)
  return iree_ok_status();
}

static iree_status_t sa_executable_lookup_export_by_name(iree_hal_executable_t* base, iree_string_view_t name,
                                                         iree_hal_executable_export_ordinal_t* out_ordinal) {
  sa_executable_t* e = sa_executable_cast(base);
  for (iree_host_size_t i = 0; i < e->export_count; ++i) {
    if (iree_string_view_equal(e->exports[i].name, name)) {
      *out_ordinal = (iree_hal_executable_export_ordinal_t)i;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND, "sa: no export '%.*s'", (int)name.size, name.data);
}

static const iree_hal_local_executable_vtable_t sa_executable_vtable = {
    .base =
        {
            .destroy = sa_executable_destroy,
            .export_count = sa_executable_export_count,
            .export_info = sa_executable_export_info,
            .export_parameters = sa_executable_export_parameters,
            .lookup_export_by_name = sa_executable_lookup_export_by_name,
        },
    .issue_call = sa_executable_issue_call,
};

static iree_status_t sa_check_range(iree_const_byte_span_t data, uint64_t offset, uint64_t bytes,
                                    const char* what) {
  if (offset + bytes > data.data_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "sa-desc-v1: %s out of the file (%llu + %llu > %llu)",
                            what, (unsigned long long)offset, (unsigned long long)bytes,
                            (unsigned long long)data.data_length);
  }
  return iree_ok_status();
}

static iree_status_t sa_executable_create(sa_context_t* context, iree_const_byte_span_t data,
                                          iree_allocator_t host, iree_hal_executable_t** out_executable) {
  IREE_RETURN_IF_ERROR(sa_check_range(data, 0, sizeof(sa_file_header_t), "header"));
  sa_file_header_t h;
  memcpy(&h, data.data, sizeof(h));
  if (memcmp(h.magic, sa_magic, 8) != 0 || h.version < 1 || h.version > 3) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "not an sa-desc executable (version 1 to 3)");
  }
  if (h.d != context->transport->d) {
    return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                            "sa-desc-v1: executable compiled for D = %u, the %s device has D = %u", h.d,
                            context->transport->name, context->transport->d);
  }
  IREE_RETURN_IF_ERROR(
      sa_check_range(data, h.export_offset, (uint64_t)h.export_count * sizeof(sa_file_export_t), "export table"));
  IREE_RETURN_IF_ERROR(sa_check_range(data, h.templates_offset, h.templates_bytes, "templates"));
  IREE_RETURN_IF_ERROR(sa_check_range(data, h.strings_offset, h.strings_bytes, "strings"));
  if (h.version >= 3) {
    IREE_RETURN_IF_ERROR(
        sa_check_range(data, h.ext_offset, (uint64_t)h.export_count * sizeof(sa_file_ext_t), "export extensions"));
  }
  if (h.export_count == 0 || h.templates_bytes == 0 || h.templates_bytes % 64) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "sa-desc-v1: empty or misaligned templates");
  }

  uint64_t setup_total = 0;
  for (uint32_t i = 0; h.version >= 2 && i < h.export_count; ++i) {
    sa_file_export_t x;
    memcpy(&x, data.data + h.export_offset + i * sizeof(x), sizeof(x));
    if (x.setup_count > 24) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "sa-desc: export %u has %u setup entries", i,
                              x.setup_count);
    }
    IREE_RETURN_IF_ERROR(sa_check_range(data, x.setup_offset, (uint64_t)x.setup_count * sizeof(sa_setup_t),
                                        "setup table"));
    setup_total += x.setup_count;
  }
  sa_executable_t* e = NULL;
  iree_host_size_t total = sizeof(*e) + h.export_count * (sizeof(sa_export_t) + sizeof(*e->attrs)) +
                           setup_total * sizeof(sa_setup_t) + h.strings_bytes;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host, total, (void**)&e));
  memset(e, 0, total);
  iree_hal_local_executable_initialize(&sa_executable_vtable, host, &e->base);
  e->context = context;
  e->export_count = h.export_count;
  e->attrs = (iree_hal_executable_dispatch_attrs_v0_t*)(e + 1);
  e->exports = (sa_export_t*)(e->attrs + h.export_count);
  sa_setup_t* setups = (sa_setup_t*)(e->exports + h.export_count);
  e->strings = (char*)(setups + setup_total);
  memcpy(e->strings, data.data + h.strings_offset, h.strings_bytes);
  e->base.dispatch_attrs = e->attrs;

  iree_status_t status = iree_allocator_malloc_aligned(sa_context_data_allocator(context), h.templates_bytes, 64,
                                                       0, &e->templates);
  uint32_t tmpl_phys = 0;
  if (iree_status_is_ok(status)) {
    memcpy(e->templates, data.data + h.templates_offset, h.templates_bytes);
    status = sa_context_phys(context, e->templates, &tmpl_phys);
  }
  for (uint32_t i = 0; iree_status_is_ok(status) && i < h.export_count; ++i) {
    sa_file_export_t x;
    memcpy(&x, data.data + h.export_offset + i * sizeof(x), sizeof(x));
    if ((uint64_t)x.name_offset + x.name_length > h.strings_bytes || x.template_offset % 64 ||
        (uint64_t)x.template_offset + 64ull * x.descriptor_count > h.templates_bytes) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "sa-desc-v1: export %u out of range", i);
      break;
    }
    uint32_t nsetup = h.version >= 2 ? x.setup_count : 0;
    if (x.binding_count > 16 || (nsetup == 0 && x.constant_count > 6)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "sa-desc: export %u has %u bindings / %u constants (at most 16 / 6 without a "
                                "setup table)", i, x.binding_count, x.constant_count);
      break;
    }
    memcpy(setups, data.data + x.setup_offset, nsetup * sizeof(sa_setup_t));
    for (uint32_t j = 0; j < nsetup; ++j) {
      const sa_setup_t* st = &setups[j];
      if (((st->kind & 15) == SA_SETUP_BASE && (st->reg > 15 || st->binding >= x.binding_count)) ||
          ((st->kind & 15) == SA_SETUP_PARAM && st->reg > 7) || (st->kind & 15) > SA_SETUP_PARAM ||
          st->constant >= (int)x.constant_count) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "sa-desc: export %u, bad setup entry %u", i, j);
        break;
      }
    }
    if (!iree_status_is_ok(status)) break;
    e->exports[i].setup_count = nsetup;
    e->exports[i].setup = setups;
    setups += nsetup;
    e->exports[i].name = iree_make_string_view(e->strings + x.name_offset, x.name_length);
    // before version 3: no prefix or head, every binding read and written, SPAD_B used
    sa_file_ext_t ext = {0, 0, 0xFFFF, 0xFFFF, 0, 1, 0, 0};
    if (h.version >= 3) memcpy(&ext, data.data + h.ext_offset + i * sizeof(ext), sizeof(ext));
    if ((uint64_t)ext.prefix_count + ext.head_count >= x.descriptor_count || ext.prefix_reg > 14) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "sa-desc: export %u, bad prefix / head", i);
      break;
    }
    uint32_t t = tmpl_phys + x.template_offset;
    e->exports[i].prefix_phys = ext.prefix_count ? t : 0;
    e->exports[i].head_phys = ext.head_count ? t + 64 * ext.prefix_count : 0;
    e->exports[i].entry_phys = t + 64 * (ext.prefix_count + ext.head_count);
    e->exports[i].prefix_reads = ext.prefix_reads;
    e->exports[i].writes = ext.writes;
    e->exports[i].reads = ext.reads;
    e->exports[i].prefix_reg = ext.prefix_reg;
    e->exports[i].spad_b = (ext.flags & 1) != 0;
    e->exports[i].descriptor_count = x.descriptor_count;
    e->exports[i].est_cycles = x.est_cycles;
    e->attrs[i].binding_count = (uint8_t)x.binding_count;
    e->attrs[i].constant_count = (uint8_t)x.constant_count;
    e->attrs[i].workgroup_size_x = e->attrs[i].workgroup_size_y = e->attrs[i].workgroup_size_z = 1;
  }
  __sync_synchronize();  // the templates are in memory before any list refers to them

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)e;
  } else {
    sa_executable_destroy((iree_hal_executable_t*)e);
  }
  return status;
}

//===----------------------------------------------------------------------===//
// sa_loader_t
//===----------------------------------------------------------------------===//

typedef struct sa_loader_t {
  iree_hal_executable_loader_t base;
  iree_allocator_t host_allocator;
  sa_context_t* context;
} sa_loader_t;

static const iree_hal_executable_loader_vtable_t sa_loader_vtable;

iree_status_t sa_loader_create(sa_context_t* context, iree_allocator_t host_allocator,
                               iree_hal_executable_loader_t** out_loader) {
  sa_loader_t* l = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*l), (void**)&l));
  iree_hal_executable_loader_initialize(&sa_loader_vtable, iree_hal_executable_import_provider_null(), &l->base);
  l->host_allocator = host_allocator;
  l->context = context;
  *out_loader = &l->base;
  return iree_ok_status();
}

static void sa_loader_destroy(iree_hal_executable_loader_t* base) {
  sa_loader_t* l = (sa_loader_t*)base;
  iree_allocator_free(l->host_allocator, l);
}

static iree_status_t sa_loader_infer_format(iree_hal_executable_loader_t* base,
                                            iree_hal_executable_caching_mode_t caching_mode,
                                            iree_const_byte_span_t data, iree_host_size_t capacity,
                                            char* out_format, iree_host_size_t* out_inferred_size) {
  if (data.data_length != 0 && data.data_length < sizeof(sa_file_header_t)) {
    return iree_make_status(IREE_STATUS_INCOMPATIBLE, "not an sa-desc-v1 executable");
  }
  sa_file_header_t h;
  memcpy(&h, data.data, sizeof(h));
  if (memcmp(h.magic, sa_magic, 8) != 0) {
    return iree_make_status(IREE_STATUS_INCOMPATIBLE, "not an sa-desc-v1 executable");
  }
  if (capacity < sizeof(SA_FORMAT)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "sa: format string capacity");
  }
  memcpy(out_format, SA_FORMAT, sizeof(SA_FORMAT));
  *out_inferred_size = (iree_host_size_t)h.strings_offset + h.strings_bytes;
  return iree_ok_status();
}

static bool sa_loader_query_support(iree_hal_executable_loader_t* base,
                                    iree_hal_executable_caching_mode_t caching_mode,
                                    iree_string_view_t format) {
  return iree_string_view_equal(format, IREE_SV(SA_FORMAT));
}

static iree_status_t sa_loader_try_load(iree_hal_executable_loader_t* base,
                                        const iree_hal_executable_params_t* params,
                                        iree_host_size_t worker_capacity, iree_hal_executable_t** out_executable) {
  sa_loader_t* l = (sa_loader_t*)base;
  return sa_executable_create(l->context, params->executable_data, l->host_allocator, out_executable);
}

static const iree_hal_executable_loader_vtable_t sa_loader_vtable = {
    .destroy = sa_loader_destroy,
    .infer_format = sa_loader_infer_format,
    .query_support = sa_loader_query_support,
    .try_load = sa_loader_try_load,
};

bool sa_executable_isa(iree_hal_executable_t* executable) {
  return iree_hal_resource_is(executable, &sa_executable_vtable);
}
