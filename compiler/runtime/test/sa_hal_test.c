// C1 test (docs/iree_compiler_plan.md §5.6): the sa HAL driver through the
// plain IREE HAL API, as the VM's HAL module drives it: driver registry ->
// device -> executable cache (sa-desc-v1) -> buffers from the device
// allocator -> a one-shot command buffer with dispatches -> queue_execute ->
// semaphore wait -> read back and compare bit for bit.
//
//   sa_hal_test <dir> [repeat]   (dir from compiler/runtime/tools/make_test_exec.py)
//
// With repeat > 0, a stress part follows: that many submissions of
// qlinear + barrier + axpb, each result checked, the timeline semaphore
// advancing by one per submission.
//
// The dispatches: "qlinear" (x0, w0 -> y0) and "axpb" (x1 -> y1, push
// constants A, B), recorded in one command buffer with a barrier between
// them, then "axpb" again in a second submission on sub-ranges of one
// buffer (binding offsets); then "fault" (a device range error: the
// submission must fail with the device status) followed by "qlinear" again
// on a fresh device-side result (the device keeps working after an error).
// Transport: SA_TRANSPORT.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "sa_context.h"

#define CHECK(expr)                                         \
  do {                                                      \
    iree_status_t s_ = (expr);                              \
    if (!iree_status_is_ok(s_)) {                           \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expr); \
      iree_status_fprint(stderr, s_);                       \
      iree_status_free(s_);                                 \
      exit(1);                                              \
    }                                                       \
  } while (0)

static void* read_file(const char* dir, const char* name, size_t* size) {
  char path[1024];
  snprintf(path, sizeof(path), "%s/%s", dir, name);
  FILE* f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "cannot open %s\n", path);
    exit(1);
  }
  fseek(f, 0, SEEK_END);
  *size = (size_t)ftell(f);
  fseek(f, 0, SEEK_SET);
  void* p = malloc(*size);
  if (fread(p, 1, *size, f) != *size) exit(1);
  fclose(f);
  return p;
}

static uint32_t read_param(const char* dir, const char* key) {
  size_t n;
  char* s = read_file(dir, "test.txt", &n);
  s = realloc(s, n + 1);
  s[n] = 0;
  for (char* line = strtok(s, "\n"); line; line = strtok(NULL, "\n")) {
    char k[32];
    unsigned long v;
    if (sscanf(line, "%31s %lu", k, &v) == 2 && strcmp(k, key) == 0) {
      free(s);
      return (uint32_t)v;
    }
  }
  fprintf(stderr, "test.txt: no %s\n", key);
  exit(1);
}

static iree_hal_buffer_t* make_buffer(iree_hal_device_t* device, size_t size, const void* init) {
  iree_hal_buffer_params_t params = {
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
      .usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER |
               IREE_HAL_BUFFER_USAGE_MAPPING,
  };
  iree_hal_buffer_t* b = NULL;
  CHECK(iree_hal_allocator_allocate_buffer(iree_hal_device_allocator(device), params, size, &b));
  if (init) {
    CHECK(iree_hal_buffer_map_write(b, 0, init, size));
  } else {
    CHECK(iree_hal_buffer_map_zero(b, 0, size));
  }
  return b;
}

static void submit_and_wait(iree_hal_device_t* device, iree_hal_command_buffer_t* cb, uint64_t* timeline,
                            iree_hal_semaphore_t* sem) {
  uint64_t signal = ++*timeline;
  iree_hal_semaphore_list_t signal_list = {1, &sem, &signal};
  CHECK(iree_hal_device_queue_execute(device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
                                      signal_list, cb, iree_hal_buffer_binding_table_empty(),
                                      IREE_HAL_EXECUTE_FLAG_NONE));
  CHECK(iree_hal_semaphore_wait(sem, signal, iree_infinite_timeout(), IREE_HAL_WAIT_FLAG_DEFAULT));
}

static int compare(const char* what, iree_hal_buffer_t* b, size_t offset, const void* want, size_t size) {
  void* got = malloc(size);
  CHECK(iree_hal_buffer_map_read(b, offset, got, size));
  size_t diff = 0;
  for (size_t i = 0; i < size / 4; ++i) diff += ((uint32_t*)got)[i] != ((const uint32_t*)want)[i];
  printf("%-28s %s (%zu of %zu words differ)\n", what, diff ? "DIFFERENT" : "bit-exact", diff, size / 4);
  free(got);
  return diff != 0;
}

int main(int argc, char** argv) {
  if (argc != 2 && argc != 3) {
    fprintf(stderr, "usage: %s <dir of make_test_exec.py> [repeat]\n", argv[0]);
    return 2;
  }
  const char* dir = argv[1];
  iree_allocator_t host = iree_allocator_system();
  uint32_t k = read_param(dir, "k"), n = read_param(dir, "n"), m = read_param(dir, "m");
  uint32_t consts[2] = {read_param(dir, "A"), read_param(dir, "B")};
  size_t exe_size, x0s, w0s, y0s, x1s, y1s;
  void* exe_data = read_file(dir, "test.sadesc", &exe_size);
  void* x0 = read_file(dir, "x0.bin", &x0s);
  void* w0 = read_file(dir, "w0.bin", &w0s);
  void* y0 = read_file(dir, "y0.bin", &y0s);
  void* x1 = read_file(dir, "x1.bin", &x1s);
  void* y1 = read_file(dir, "y1.bin", &y1s);
  if (x0s != 4 * k || y0s != 4 * n || x1s != 4 * m || y1s != 4 * m) {
    fprintf(stderr, "test data sizes do not match test.txt\n");
    return 1;
  }

  // driver and device
  iree_hal_driver_registry_t* registry = NULL;
  CHECK(iree_hal_driver_registry_allocate(host, &registry));
  CHECK(iree_hal_sa_driver_module_register(registry));
  iree_hal_driver_t* driver = NULL;
  CHECK(iree_hal_driver_registry_try_create(registry, IREE_SV("sa"), host, &driver));
  iree_hal_device_t* device = NULL;
  CHECK(iree_hal_driver_create_default_device(driver, host, &device));
  sa_context_t* ctx = NULL;
  CHECK(sa_context_get(&ctx));
  printf("sa device: transport %s, D = %u, window %u MB at %#x\n", ctx->transport->name, ctx->transport->d,
         ctx->transport->mem_size >> 20, ctx->transport->mem_phys);

  // executable
  iree_hal_executable_cache_t* cache = NULL;
  iree_status_t loop_status = iree_ok_status();
  CHECK(iree_hal_executable_cache_create(device, IREE_SV("default"), iree_loop_inline(&loop_status), &cache));
  iree_hal_executable_params_t params;
  iree_hal_executable_params_initialize(&params);
  params.caching_mode = IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA;
  params.executable_format = IREE_SV("sa-desc-v1");
  params.executable_data = iree_make_const_byte_span(exe_data, exe_size);
  iree_hal_executable_t* exe = NULL;
  CHECK(iree_hal_executable_cache_prepare_executable(cache, &params, &exe));
  iree_hal_executable_export_ordinal_t qlinear = 0, axpb = 0;
  CHECK(iree_hal_executable_lookup_export_by_name(exe, IREE_SV("qlinear"), &qlinear));
  CHECK(iree_hal_executable_lookup_export_by_name(exe, IREE_SV("axpb"), &axpb));
  printf("executable: %zu exports (qlinear = %u, axpb = %u)\n", iree_hal_executable_export_count(exe), qlinear,
         axpb);

  // buffers
  iree_hal_buffer_t* bx0 = make_buffer(device, x0s, x0);
  iree_hal_buffer_t* bw0 = make_buffer(device, w0s, w0);
  iree_hal_buffer_t* by0 = make_buffer(device, y0s, NULL);
  iree_hal_buffer_t* bx1 = make_buffer(device, x1s, x1);
  iree_hal_buffer_t* by1 = make_buffer(device, y1s, NULL);

  iree_hal_semaphore_t* sem = NULL;
  CHECK(iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY, 0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &sem));
  uint64_t timeline = 0;
  int fails = 0;

  // submission 1: qlinear, barrier, axpb
  iree_hal_command_buffer_t* cb = NULL;
  CHECK(iree_hal_command_buffer_create(device, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
                                       IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY, 0, &cb));
  CHECK(iree_hal_command_buffer_begin(cb));
  iree_hal_buffer_ref_t q_refs[3] = {iree_hal_make_buffer_ref(bx0, 0, x0s), iree_hal_make_buffer_ref(bw0, 0, w0s),
                                     iree_hal_make_buffer_ref(by0, 0, y0s)};
  CHECK(iree_hal_command_buffer_dispatch(cb, exe, qlinear, iree_hal_make_static_dispatch_config(1, 1, 1),
                                         iree_const_byte_span_empty(), (iree_hal_buffer_ref_list_t){3, q_refs},
                                         IREE_HAL_DISPATCH_FLAG_NONE));
  CHECK(iree_hal_command_buffer_execution_barrier(cb, IREE_HAL_EXECUTION_STAGE_DISPATCH,
                                                  IREE_HAL_EXECUTION_STAGE_DISPATCH,
                                                  IREE_HAL_EXECUTION_BARRIER_FLAG_NONE, 0, NULL, 0, NULL));
  iree_hal_buffer_ref_t a_refs[2] = {iree_hal_make_buffer_ref(bx1, 0, x1s), iree_hal_make_buffer_ref(by1, 0, y1s)};
  CHECK(iree_hal_command_buffer_dispatch(cb, exe, axpb, iree_hal_make_static_dispatch_config(1, 1, 1),
                                         iree_make_const_byte_span(consts, sizeof(consts)),
                                         (iree_hal_buffer_ref_list_t){2, a_refs}, IREE_HAL_DISPATCH_FLAG_NONE));
  CHECK(iree_hal_command_buffer_end(cb));
  submit_and_wait(device, cb, &timeline, sem);
  iree_hal_command_buffer_release(cb);
  fails += compare("qlinear (x0, w0 -> y0)", by0, 0, y0, y0s);
  fails += compare("axpb (x1 -> y1)", by1, 0, y1, y1s);

  // submission 2: axpb on sub-ranges of one buffer (binding offsets): [pad | x1 | pad | y]
  size_t pad = 4096;
  iree_hal_buffer_t* big = make_buffer(device, 2 * pad + x1s + y1s, NULL);
  CHECK(iree_hal_buffer_map_write(big, pad, x1, x1s));
  CHECK(iree_hal_command_buffer_create(device, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
                                       IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY, 0, &cb));
  CHECK(iree_hal_command_buffer_begin(cb));
  iree_hal_buffer_ref_t o_refs[2] = {iree_hal_make_buffer_ref(big, pad, x1s),
                                     iree_hal_make_buffer_ref(big, 2 * pad + x1s, y1s)};
  CHECK(iree_hal_command_buffer_dispatch(cb, exe, axpb, iree_hal_make_static_dispatch_config(1, 1, 1),
                                         iree_make_const_byte_span(consts, sizeof(consts)),
                                         (iree_hal_buffer_ref_list_t){2, o_refs}, IREE_HAL_DISPATCH_FLAG_NONE));
  CHECK(iree_hal_command_buffer_end(cb));
  submit_and_wait(device, cb, &timeline, sem);
  iree_hal_command_buffer_release(cb);
  fails += compare("axpb (binding offsets)", big, 2 * pad + x1s, y1, y1s);

  // submission 3: the device reports an error -> the submission fails
  iree_hal_executable_export_ordinal_t fault = 0;
  CHECK(iree_hal_executable_lookup_export_by_name(exe, IREE_SV("fault"), &fault));
  CHECK(iree_hal_command_buffer_create(device, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
                                       IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY, 0, &cb));
  CHECK(iree_hal_command_buffer_begin(cb));
  CHECK(iree_hal_command_buffer_dispatch(cb, exe, fault, iree_hal_make_static_dispatch_config(1, 1, 1),
                                         iree_const_byte_span_empty(), (iree_hal_buffer_ref_list_t){1, q_refs},
                                         IREE_HAL_DISPATCH_FLAG_NONE));
  CHECK(iree_hal_command_buffer_end(cb));
  iree_hal_semaphore_t* sem2 = NULL;
  CHECK(iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY, 0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &sem2));
  uint64_t one = 1;
  iree_hal_semaphore_list_t sig2 = {1, &sem2, &one};
  iree_status_t st = iree_hal_device_queue_execute(device, IREE_HAL_QUEUE_AFFINITY_ANY,
                                                   iree_hal_semaphore_list_empty(), sig2, cb,
                                                   iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE);
  if (iree_status_is_ok(st)) st = iree_hal_semaphore_wait(sem2, 1, iree_infinite_timeout(), IREE_HAL_WAIT_FLAG_DEFAULT);
  iree_hal_command_buffer_release(cb);
  iree_hal_semaphore_release(sem2);
  if (iree_status_is_ok(st)) {
    printf("%-28s NOT REPORTED\n", "fault (device error)");
    fails++;
  } else {
    char msg[512];
    iree_host_size_t len = 0;
    iree_status_format(st, sizeof(msg), msg, &len);
    char* line = strstr(msg, "sa: dispatch 'fault' failed");
    printf("%-28s reported: %.*s\n", "fault (device error)", line ? (int)strcspn(line, "\n;") : 60,
           line ? line : msg);
    if (!line) fails++;
    iree_status_free(st);
  }

  // submission 4: the device still works (qlinear into a cleared y0)
  CHECK(iree_hal_buffer_map_zero(by0, 0, y0s));
  CHECK(iree_hal_command_buffer_create(device, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
                                       IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY, 0, &cb));
  CHECK(iree_hal_command_buffer_begin(cb));
  CHECK(iree_hal_command_buffer_dispatch(cb, exe, qlinear, iree_hal_make_static_dispatch_config(1, 1, 1),
                                         iree_const_byte_span_empty(), (iree_hal_buffer_ref_list_t){3, q_refs},
                                         IREE_HAL_DISPATCH_FLAG_NONE));
  CHECK(iree_hal_command_buffer_end(cb));
  submit_and_wait(device, cb, &timeline, sem);
  iree_hal_command_buffer_release(cb);
  fails += compare("qlinear after the fault", by0, 0, y0, y0s);

  // stress: repeated submissions
  int repeat = argc == 3 ? atoi(argv[2]) : 0;
  if (repeat > 0) {
    void* got0 = malloc(y0s);
    void* got1 = malloc(y1s);
    int bad = 0;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < repeat; ++r) {
      CHECK(iree_hal_buffer_map_zero(by0, 0, y0s));
      CHECK(iree_hal_buffer_map_zero(by1, 0, y1s));
      CHECK(iree_hal_command_buffer_create(device, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
                                           IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY, 0, &cb));
      CHECK(iree_hal_command_buffer_begin(cb));
      CHECK(iree_hal_command_buffer_dispatch(cb, exe, qlinear, iree_hal_make_static_dispatch_config(1, 1, 1),
                                             iree_const_byte_span_empty(), (iree_hal_buffer_ref_list_t){3, q_refs},
                                             IREE_HAL_DISPATCH_FLAG_NONE));
      CHECK(iree_hal_command_buffer_execution_barrier(cb, IREE_HAL_EXECUTION_STAGE_DISPATCH,
                                                      IREE_HAL_EXECUTION_STAGE_DISPATCH,
                                                      IREE_HAL_EXECUTION_BARRIER_FLAG_NONE, 0, NULL, 0, NULL));
      CHECK(iree_hal_command_buffer_dispatch(cb, exe, axpb, iree_hal_make_static_dispatch_config(1, 1, 1),
                                             iree_make_const_byte_span(consts, sizeof(consts)),
                                             (iree_hal_buffer_ref_list_t){2, a_refs}, IREE_HAL_DISPATCH_FLAG_NONE));
      CHECK(iree_hal_command_buffer_end(cb));
      submit_and_wait(device, cb, &timeline, sem);
      iree_hal_command_buffer_release(cb);
      CHECK(iree_hal_buffer_map_read(by0, 0, got0, y0s));
      CHECK(iree_hal_buffer_map_read(by1, 0, got1, y1s));
      bad += memcmp(got0, y0, y0s) != 0 || memcmp(got1, y1, y1s) != 0;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double us = ((double)(t1.tv_sec - t0.tv_sec) * 1e6 + (double)(t1.tv_nsec - t0.tv_nsec) / 1e3) / repeat;
    uint64_t value = 0;
    CHECK(iree_hal_semaphore_query(sem, &value));
    printf("%-28s %d submissions, %d wrong, semaphore %llu (expected %llu), %.0f us per submission\n",
           "stress (qlinear + axpb)", repeat, bad, (unsigned long long)value, (unsigned long long)timeline, us);
    fails += bad != 0 || value != timeline;
    free(got0);
    free(got1);
  }

  printf("dispatch lists run: %llu\n", (unsigned long long)ctx->dispatches);
  printf("%s\n", fails ? "C1 HAL test FAILED" : "C1 HAL test PASS");

  iree_hal_buffer_release(big);
  iree_hal_buffer_release(bx0);
  iree_hal_buffer_release(bw0);
  iree_hal_buffer_release(by0);
  iree_hal_buffer_release(bx1);
  iree_hal_buffer_release(by1);
  iree_hal_semaphore_release(sem);
  iree_hal_executable_release(exe);
  iree_hal_executable_cache_release(cache);
  iree_hal_device_release(device);
  iree_hal_driver_release(driver);
  iree_hal_driver_registry_free(registry);
  return fails ? 1 : 0;
}
