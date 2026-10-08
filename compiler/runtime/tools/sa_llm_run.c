// sa-llm-run: decode loop of an IREE-compiled llama (compiler/frontend/qllama.py)
// on any IREE device, in one process so that the module's KV cache (mutable
// globals) persists across tokens (docs/iree_compiler_plan.md §6.8 G): the
// IREE counterpart of llm/runtime.py LlamaDevice.
//
//   sa-llm-run --device=sa --module=sa.vmfb --parameters=model=sa_packed.irpa \
//       --tokens=1,9038,2501 [--generate=20] [--stop_token=2] [--valid_len=256] [--logits_out=logits.f32]
//
// Each step calls main(token: i64[1], pos: i64[1], valid: f32[valid_len]) with
// valid = 1.0 for positions 0..pos (the attention reads positions by pos; the
// sa form has a static valid_len = seq_len). The tokens are fed in order; with
// --generate the next tokens are the argmax of the logits (greedy). Every
// step's logits (float32) are appended to --logits_out. Prints the tokens and
// the time per step.
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/types.h"
#include "iree/tooling/context_util.h"
#include "iree/vm/api.h"
#include "sa_context.h"

IREE_FLAG(string, tokens, "1", "Comma-separated token ids fed in order.");
IREE_FLAG(int32_t, generate, 0, "Then generate this many tokens greedily.");
IREE_FLAG(int32_t, valid_len, 256, "Length of the valid input (the static attention length).");
IREE_FLAG(int32_t, pad, 0,
          "Dynamic attention length: valid has length pos + 1 rounded up to a multiple of this (0: valid_len).");
IREE_FLAG(string, logits_out, "", "File the float32 logits of every step are appended to.");
IREE_FLAG(int32_t, stop_token, -1, "Stop generating after this token (an end-of-sequence id; -1: none).");
IREE_FLAG(string, abi, "sa",
          "Calling convention of main: \"sa\" main(token[1], pos[1], valid) (the frontend's QLlama / QModel), "
          "\"hf\" main(input_ids[1,1], position_ids[1,1]) (an unmodified HuggingFace decoder, plan §8.16).");
IREE_FLAG(int32_t, prefill, 0,
          "Prefill chunk M: a prompt of >= M tokens through the module's prefill(tokens[M], positions[M], valid) "
          "in chunks at 0, M, 2M, ... and P - M, then decode (plan §8.13).");

// The device clock for cycles -> ms: SA_DEVICE_HZ (the launcher passes the
// overlay's, e.g. 100 MHz on the KV260's K1b), else the PYNQ-Z1's 50 MHz.
static double sa_device_hz(void) {
  const char* e = getenv("SA_DEVICE_HZ");
  double hz = e ? atof(e) : 0.0;
  return hz > 0.0 ? hz : 50e6;
}

static iree_status_t make_view(iree_hal_device_t* device, iree_hal_allocator_t* allocator, const void* data,
                               iree_host_size_t n, iree_hal_element_type_t type, iree_hal_buffer_view_t** out) {
  iree_hal_dim_t shape[2] = {n, 1};
  // --abi=hf: ids / positions as [1, n]
  iree_host_size_t rank = (strcmp(FLAG_abi, "hf") == 0 && type == IREE_HAL_ELEMENT_TYPE_INT_64) ? 2 : 1;
  if (rank == 2) shape[0] = 1, shape[1] = n;
  iree_hal_buffer_params_t params = {
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
  };
  return iree_hal_buffer_view_allocate_buffer_copy(
      device, allocator, rank, shape, type, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, params,
      iree_make_const_byte_span(data, n * iree_hal_element_dense_byte_count(type)), out);
}

// f(a[na]: i64, b[nb]: i64, valid[nv]: f32) -> logits (f32, copied to *logits);
// --abi=hf: f(a[1, na], b[1, nb])
static iree_status_t call3(iree_vm_context_t* context, iree_vm_function_t function, iree_hal_device_t* device,
                           iree_hal_allocator_t* allocator, iree_allocator_t host, const int64_t* a, int na,
                           const int64_t* b, int nb, const float* valid, iree_host_size_t nv, float** logits,
                           iree_host_size_t* vocab) {
  iree_vm_list_t *inputs = NULL, *outputs = NULL;
  iree_hal_buffer_view_t *ba = NULL, *bbv = NULL, *bv = NULL;
  iree_status_t status = iree_vm_list_create(iree_vm_make_undefined_type_def(), 3, host, &inputs);
  bool hf = strcmp(FLAG_abi, "hf") == 0;             // (a[1, na], b[1, nb]): no valid
  if (iree_status_is_ok(status)) status = make_view(device, allocator, a, na, IREE_HAL_ELEMENT_TYPE_INT_64, &ba);
  if (iree_status_is_ok(status)) status = make_view(device, allocator, b, nb, IREE_HAL_ELEMENT_TYPE_INT_64, &bbv);
  if (iree_status_is_ok(status) && !hf)
    status = make_view(device, allocator, valid, nv, IREE_HAL_ELEMENT_TYPE_FLOAT_32, &bv);
  iree_hal_buffer_view_t* views[3] = {ba, bbv, bv};
  for (int i = 0; i < (hf ? 2 : 3) && iree_status_is_ok(status); ++i) {
    iree_vm_ref_t r = iree_hal_buffer_view_move_ref(views[i]);
    views[i] = NULL;
    status = iree_vm_list_push_ref_move(inputs, &r);
  }
  for (int i = 0; i < 3; ++i) iree_hal_buffer_view_release(views[i]);
  if (iree_status_is_ok(status)) status = iree_vm_list_create(iree_vm_make_undefined_type_def(), 1, host, &outputs);
  if (iree_status_is_ok(status))
    status = iree_vm_invoke(context, function, IREE_VM_INVOCATION_FLAG_NONE, NULL, inputs, outputs, host);
  iree_hal_buffer_view_t* out = NULL;
  if (iree_status_is_ok(status)) {
    out = iree_vm_list_get_buffer_view_assign(outputs, 0);
    if (!out) status = iree_make_status(IREE_STATUS_INTERNAL, "no output");
  }
  if (iree_status_is_ok(status)) {
    *vocab = iree_hal_buffer_view_element_count(out);
    if (!*logits) *logits = malloc(*vocab * sizeof(float));
    status = iree_hal_device_transfer_d2h(device, iree_hal_buffer_view_buffer(out), 0, *logits, *vocab * sizeof(float),
                                          IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
  }
  iree_vm_list_release(inputs);
  iree_vm_list_release(outputs);
  return status;
}

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static iree_status_t run(iree_allocator_t host) {
  iree_vm_instance_t* instance = NULL;
  IREE_RETURN_IF_ERROR(iree_tooling_create_instance(host, &instance));
  iree_tooling_module_list_t modules;
  iree_tooling_module_list_initialize(&modules);
  iree_status_t status = iree_tooling_load_modules_from_flags(instance, host, &modules);
  iree_vm_context_t* context = NULL;
  iree_hal_device_t* device = NULL;
  iree_hal_allocator_t* allocator = NULL;
  iree_vm_module_t* main_module = NULL;
  if (iree_status_is_ok(status)) {
    main_module = iree_tooling_module_list_back(&modules);
    if (!main_module) status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "no --module given");
  }
  if (iree_status_is_ok(status)) {
    status = iree_tooling_create_context_from_flags(instance, modules.count, modules.values, IREE_SV("sa"), host,
                                                    &context, &device, &allocator);
  }
  iree_vm_function_t function;
  if (iree_status_is_ok(status)) {
    status = iree_vm_module_lookup_function_by_name(main_module, IREE_VM_FUNCTION_LINKAGE_EXPORT, IREE_SV("main"),
                                                    &function);
  }
  FILE* lf = NULL;
  if (iree_status_is_ok(status) && FLAG_logits_out[0]) {
    lf = fopen(FLAG_logits_out, "wb");
    if (!lf) status = iree_make_status(IREE_STATUS_UNAVAILABLE, "cannot open %s", FLAG_logits_out);
  }

  // the prompt tokens
  int64_t toks[4096];
  int ntok = 0;
  for (const char* p = FLAG_tokens; *p && ntok < 4096;) {
    toks[ntok++] = strtoll(p, (char**)&p, 10);
    while (*p == ',' || *p == ' ') ++p;
  }
  int steps = ntok + FLAG_generate;
  if (steps > FLAG_valid_len) steps = FLAG_valid_len;
  float* valid = calloc((size_t)FLAG_valid_len, sizeof(float));
  float* logits = NULL;
  iree_host_size_t vocab = 0;
  double total = 0;
  int64_t token = toks[0];
  printf("tokens:");
  // prefill: the prompt in chunks of M (the last one at P - M), its last
  // position's logits give the first generated token
  int pos0 = 0, M = FLAG_prefill;
  double prefill_time = 0;
  int chunks = 0;
  if (iree_status_is_ok(status) && M > 0 && ntok >= M) {
    iree_vm_function_t pf;
    status = iree_vm_module_lookup_function_by_name(main_module, IREE_VM_FUNCTION_LINKAGE_EXPORT, IREE_SV("prefill"),
                                                    &pf);
    // the chunks before the last through prefill_kv (no classifier) when the module has it
    iree_vm_function_t pkv = pf;
    if (iree_status_is_ok(status)) {
      iree_status_t st = iree_vm_module_lookup_function_by_name(main_module, IREE_VM_FUNCTION_LINKAGE_EXPORT,
                                                                IREE_SV("prefill_kv"), &pkv);
      if (!iree_status_is_ok(st)) {
        iree_status_ignore(st);
        pkv = pf;
      }
    }
    float* kv_out = NULL;
    iree_host_size_t kv_n = 0;
    int64_t* pos_buf = malloc((size_t)M * sizeof(int64_t));
    for (int s0 = 0; iree_status_is_ok(status);) {
      int s = s0 + M >= ntok ? ntok - M : s0;
      for (int i = 0; i < M; ++i) pos_buf[i] = s + i;
      for (int i = 0; i <= s + M - 1; ++i) valid[i] = 1.0f;
      iree_host_size_t vlen = FLAG_pad > 0 ? (iree_host_size_t)(((s + M - 1) / FLAG_pad + 1) * FLAG_pad)
                                           : (iree_host_size_t)FLAG_valid_len;
      double t0 = now();
      if (s + M >= ntok)
        status = call3(context, pf, device, allocator, host, toks + s, M, pos_buf, M, valid, vlen, &logits, &vocab);
      else
        status = call3(context, pkv, device, allocator, host, toks + s, M, pos_buf, M, valid, vlen, &kv_out, &kv_n);
      prefill_time += now() - t0;
      ++chunks;
      if (chunks == 1) sa_context_profile_reset();   // the first chunk includes loading
      if (s + M >= ntok) break;
      s0 += M;
    }
    free(pos_buf);
    free(kv_out);
    if (iree_status_is_ok(status) && chunks > 1 && getenv("SA_PROFILE")) {
      uint64_t nd, cyc, ns;
      sa_context_profile_totals(&nd, &cyc, &ns);
      printf("prefill per chunk (after the first, %d chunks of %d): %.1f %s, device %.2f ms (%.0f cycles at %.1f MHz), "
             "submissions %.2f ms\n",
             chunks - 1, M, nd / (double)(chunks - 1), strcmp(getenv("SA_PROFILE"), "batch") ? "dispatches" : "lists",
             cyc / (chunks - 1) / (sa_device_hz() / 1e3), cyc / (double)(chunks - 1), sa_device_hz() / 1e6,
             ns / 1e6 / (chunks - 1));
      sa_context_profile_report(stdout, chunks - 1);
    }
    if (iree_status_is_ok(status)) {
      if (lf) fwrite(logits, sizeof(float), vocab, lf);
      iree_host_size_t best = 0;
      for (iree_host_size_t i = 1; i < vocab; ++i)
        if (logits[i] > logits[best]) best = i;
      for (int i = 0; i < ntok; ++i) printf(" %" PRId64, toks[i]);
      fflush(stdout);
      token = (int64_t)best;
      pos0 = ntok;
      if (FLAG_stop_token >= 0 && token == FLAG_stop_token) steps = ntok;
      sa_context_profile_reset();
    }
  }
  for (int pos = pos0; pos < steps && iree_status_is_ok(status); ++pos) {
    if (pos < ntok) token = toks[pos];
    valid[pos] = 1.0f;
    int64_t pos64 = pos;
    iree_vm_list_t *inputs = NULL, *outputs = NULL;
    status = iree_vm_list_create(iree_vm_make_undefined_type_def(), 3, host, &inputs);
    iree_hal_buffer_view_t *bt = NULL, *bp = NULL, *bv = NULL;
    if (iree_status_is_ok(status)) status = make_view(device, allocator, &token, 1, IREE_HAL_ELEMENT_TYPE_INT_64, &bt);
    if (iree_status_is_ok(status)) status = make_view(device, allocator, &pos64, 1, IREE_HAL_ELEMENT_TYPE_INT_64, &bp);
    iree_host_size_t vlen = FLAG_pad > 0 ? (iree_host_size_t)((pos / FLAG_pad + 1) * FLAG_pad) : (iree_host_size_t)FLAG_valid_len;
    bool hf = strcmp(FLAG_abi, "hf") == 0;
    if (iree_status_is_ok(status) && !hf)
      status = make_view(device, allocator, valid, vlen, IREE_HAL_ELEMENT_TYPE_FLOAT_32, &bv);
    if (iree_status_is_ok(status)) {
      iree_vm_ref_t r = iree_hal_buffer_view_move_ref(bt);
      status = iree_vm_list_push_ref_move(inputs, &r);
    }
    if (iree_status_is_ok(status)) {
      iree_vm_ref_t r = iree_hal_buffer_view_move_ref(bp);
      status = iree_vm_list_push_ref_move(inputs, &r);
    }
    if (iree_status_is_ok(status) && !hf) {
      iree_vm_ref_t r = iree_hal_buffer_view_move_ref(bv);
      status = iree_vm_list_push_ref_move(inputs, &r);
    }
    if (iree_status_is_ok(status)) status = iree_vm_list_create(iree_vm_make_undefined_type_def(), 1, host, &outputs);
    double t0 = now();
    if (iree_status_is_ok(status))
      status = iree_vm_invoke(context, function, IREE_VM_INVOCATION_FLAG_NONE, NULL, inputs, outputs, host);
    double dt = now() - t0;
    if (pos == 0) sa_context_profile_reset();      // the first step includes loading
    iree_hal_buffer_view_t* out = NULL;
    if (iree_status_is_ok(status)) {
      out = iree_vm_list_get_buffer_view_assign(outputs, 0);
      if (!out) status = iree_make_status(IREE_STATUS_INTERNAL, "no output");
    }
    if (iree_status_is_ok(status)) {
      vocab = iree_hal_buffer_view_element_count(out);
      if (!logits) logits = malloc(vocab * sizeof(float));
      status = iree_hal_device_transfer_d2h(device, iree_hal_buffer_view_buffer(out), 0, logits,
                                            vocab * sizeof(float), IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
                                            iree_infinite_timeout());
    }
    if (iree_status_is_ok(status)) {
      if (lf) fwrite(logits, sizeof(float), vocab, lf);
      iree_host_size_t best = 0;
      for (iree_host_size_t i = 1; i < vocab; ++i)
        if (logits[i] > logits[best]) best = i;
      if (pos > 0) total += dt;
      printf(" %" PRId64, token);
      fflush(stdout);
      token = (int64_t)best;                 // next input when generating
      if (pos + 1 >= ntok && FLAG_stop_token >= 0 && token == FLAG_stop_token) steps = pos + 1;
    }
    iree_vm_list_release(inputs);
    iree_vm_list_release(outputs);
  }
  if (iree_status_is_ok(status) && FLAG_generate) printf(" %" PRId64, token);
  printf("\n");
  if (iree_status_is_ok(status) && pos0 > 0) {
    printf("prefill: %d prompt tokens in %d chunks of %d, %.1f ms (%.2f tokens/s; the first chunk includes loading)",
           ntok, chunks, M, 1e3 * prefill_time, ntok / prefill_time);
    if (steps > pos0)
      printf("; %d decode steps, %.1f ms per step (%.2f tokens/s)", steps - pos0, 1e3 * total / (steps - pos0),
             (steps - pos0) / total);
    printf("\n");
  } else if (iree_status_is_ok(status) && steps > 1) {
    printf("%d steps, %.1f ms per step after the first (%.2f tokens/s)\n", steps, 1e3 * total / (steps - 1),
           (steps - 1) / total);
  }
  if (iree_status_is_ok(status)) {
    sa_context_t* sc = NULL;
    if (iree_status_is_ok(sa_context_get(&sc)))
      printf("device memory: peak %.1f MB of the %.1f MB heap\n", sc->heap_peak / 1048576.0,
             (double)(sc->heap_end - sc->heap_begin) / 1048576.0);
  }
  // the steps profiled: decode after the prefill (the profile is reset after
  // it), else every step after the first
  int np = pos0 > 0 ? steps - pos0 : steps - 1;
  if (iree_status_is_ok(status) && np > 0 && getenv("SA_PROFILE")) {
    uint64_t nd, cyc, ns;
    sa_context_profile_totals(&nd, &cyc, &ns);
    printf("per step (%s): %.1f %s, device %.2f ms (%.0f cycles at %.1f MHz), "
           "submissions %.2f ms, rest of the runtime %.2f ms\n",
           pos0 > 0 ? "decode after the prefill" : "after the first", nd / (double)np,
           strcmp(getenv("SA_PROFILE"), "batch") ? "dispatches" : "lists", cyc / np / (sa_device_hz() / 1e3), cyc / (double)np, sa_device_hz() / 1e6,
           ns / 1e6 / np, 1e3 * total / np - ns / 1e6 / np);
    sa_context_t* sc = NULL;
    if (iree_status_is_ok(sa_context_get(&sc)))
      printf("per step: %.1f barriers -> %.1f FENCEs; prefixes run inside an earlier dispatch %.1f, before "
             "their FENCE %.1f\n",
             sc->batch_barriers / (double)steps, sc->batch_fences / (double)steps, sc->batch_hosted / (double)steps,
             sc->batch_hoisted / (double)steps);
    sa_context_profile_report(stdout, np);
  }
  if (lf) fclose(lf);
  free(valid);
  free(logits);
  iree_vm_context_release(context);
  iree_hal_allocator_release(allocator);
  iree_hal_device_release(device);
  iree_tooling_module_list_reset(&modules);
  iree_vm_instance_release(instance);
  return status;
}

int main(int argc, char** argv) {
  iree_flags_set_usage("sa-llm-run", "Decode loop of an IREE-compiled llama (qllama.py).\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  iree_status_t status = run(iree_allocator_system());
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return 1;
  }
  return 0;
}
