// sa-llm-run: decode loop of an IREE-compiled llama (compiler/frontend/qllama.py)
// on any IREE device, in one process so that the module's KV cache (mutable
// globals) persists across tokens (docs/iree_compiler_plan.md §6.8 G): the
// IREE counterpart of llm/runtime.py LlamaDevice.
//
//   sa-llm-run --device=sa --module=sa.vmfb --parameters=model=sa_packed.irpa \
//       --tokens=1,9038,2501 [--generate=20] [--valid_len=256] [--logits_out=logits.f32]
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

static iree_status_t make_view(iree_hal_device_t* device, iree_hal_allocator_t* allocator, const void* data,
                               iree_host_size_t n, iree_hal_element_type_t type, iree_hal_buffer_view_t** out) {
  iree_hal_dim_t shape[1] = {n};
  iree_hal_buffer_params_t params = {
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
  };
  return iree_hal_buffer_view_allocate_buffer_copy(
      device, allocator, 1, shape, type, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, params,
      iree_make_const_byte_span(data, n * iree_hal_element_dense_byte_count(type)), out);
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
  for (int pos = 0; pos < steps && iree_status_is_ok(status); ++pos) {
    if (pos < ntok) token = toks[pos];
    valid[pos] = 1.0f;
    int64_t pos64 = pos;
    iree_vm_list_t *inputs = NULL, *outputs = NULL;
    status = iree_vm_list_create(iree_vm_make_undefined_type_def(), 3, host, &inputs);
    iree_hal_buffer_view_t *bt = NULL, *bp = NULL, *bv = NULL;
    if (iree_status_is_ok(status)) status = make_view(device, allocator, &token, 1, IREE_HAL_ELEMENT_TYPE_INT_64, &bt);
    if (iree_status_is_ok(status)) status = make_view(device, allocator, &pos64, 1, IREE_HAL_ELEMENT_TYPE_INT_64, &bp);
    iree_host_size_t vlen = FLAG_pad > 0 ? (iree_host_size_t)((pos / FLAG_pad + 1) * FLAG_pad) : (iree_host_size_t)FLAG_valid_len;
    if (iree_status_is_ok(status))
      status = make_view(device, allocator, valid, vlen, IREE_HAL_ELEMENT_TYPE_FLOAT_32, &bv);
    if (iree_status_is_ok(status)) {
      iree_vm_ref_t r = iree_hal_buffer_view_move_ref(bt);
      status = iree_vm_list_push_ref_move(inputs, &r);
    }
    if (iree_status_is_ok(status)) {
      iree_vm_ref_t r = iree_hal_buffer_view_move_ref(bp);
      status = iree_vm_list_push_ref_move(inputs, &r);
    }
    if (iree_status_is_ok(status)) {
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
    }
    iree_vm_list_release(inputs);
    iree_vm_list_release(outputs);
  }
  if (iree_status_is_ok(status) && FLAG_generate) printf(" %" PRId64, token);
  printf("\n");
  if (iree_status_is_ok(status) && steps > 1)
    printf("%d steps, %.1f ms per step after the first (%.2f tokens/s)\n", steps, 1e3 * total / (steps - 1),
           (steps - 1) / total);
  if (iree_status_is_ok(status) && steps > 1 && getenv("SA_PROFILE")) {
    uint64_t nd, cyc, ns;
    sa_context_profile_totals(&nd, &cyc, &ns);
    printf("per step (after the first): %.1f %s, device %.2f ms (%.0f cycles at 50 MHz), "
           "submissions %.2f ms, rest of the runtime %.2f ms\n",
           nd / (double)(steps - 1), strcmp(getenv("SA_PROFILE"), "batch") ? "dispatches" : "lists", cyc / (steps - 1) / 50e3, cyc / (double)(steps - 1),
           ns / 1e6 / (steps - 1), 1e3 * total / (steps - 1) - ns / 1e6 / (steps - 1));
    sa_context_t* sc = NULL;
    if (iree_status_is_ok(sa_context_get(&sc)))
      printf("per step: %.1f barriers -> %.1f FENCEs; prefixes run inside an earlier dispatch %.1f, before "
             "their FENCE %.1f\n",
             sc->batch_barriers / (double)steps, sc->batch_fences / (double)steps, sc->batch_hosted / (double)steps,
             sc->batch_hoisted / (double)steps);
    sa_context_profile_report(stdout, steps - 1);
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
