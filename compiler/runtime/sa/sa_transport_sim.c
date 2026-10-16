// sa HAL driver: the `sim` transport (docs/iree_compiler_plan.md §5.5).
//
// Talks to compiler/sim/sa_sim_server.py over a Unix socket (SA_SIM_SOCKET,
// default /tmp/sa_sim.sock). The server's shared-memory file is the device's
// DDR window; this process maps the same file, so buffers written here are
// what the functional simulator reads, and its results appear here.
//   <- HELLO <D> <shm path> <physical base> <bytes>
//   -> RUN <list physical address> [<dispatch name>]  (single-dispatch lists: the export)
//   <- DONE <status> <cycles> <descriptors> <end value>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include "sa_context.h"

typedef struct sa_sim_t {
  int fd;
  char buf[512];
  size_t len;
} sa_sim_t;

static iree_status_t sa_sim_readline(sa_sim_t* s, char* line, size_t cap) {
  for (;;) {
    char* nl = memchr(s->buf, '\n', s->len);
    if (nl) {
      size_t n = (size_t)(nl - s->buf);
      if (n >= cap) n = cap - 1;
      memcpy(line, s->buf, n);
      line[n] = 0;
      size_t used = (size_t)(nl - s->buf) + 1;
      memmove(s->buf, s->buf + used, s->len - used);
      s->len -= used;
      return iree_ok_status();
    }
    if (s->len == sizeof(s->buf)) {
      return iree_make_status(IREE_STATUS_DATA_LOSS, "sa sim: line too long");
    }
    ssize_t r = read(s->fd, s->buf + s->len, sizeof(s->buf) - s->len);
    if (r <= 0) return iree_make_status(IREE_STATUS_UNAVAILABLE, "sa sim: the simulator closed the connection");
    s->len += (size_t)r;
  }
}

static iree_status_t sa_sim_run(sa_transport_t* t, uint32_t list_phys, sa_completion_t* out) {
  sa_sim_t* s = (sa_sim_t*)t->impl;
  char line[256];
  int n = t->tag.size ? snprintf(line, sizeof(line), "RUN 0x%08x %.*s\n", list_phys, (int)(t->tag.size < 200 ? t->tag.size : 200), t->tag.data)
                       : snprintf(line, sizeof(line), "RUN 0x%08x\n", list_phys);
  if (write(s->fd, line, (size_t)n) != n) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE, "sa sim: write to the simulator failed");
  }
  IREE_RETURN_IF_ERROR(sa_sim_readline(s, line, sizeof(line)));
  if (sscanf(line, "DONE %u %u %u %u", &out->status, &out->cycles, &out->descriptors, &out->end) != 4) {
    return iree_make_status(IREE_STATUS_DATA_LOSS, "sa sim: unexpected reply '%s'", line);
  }
  return iree_ok_status();
}

#if defined(__x86_64__)
// Generated RISC-V code on the sim (plan A, sa_rv.c): the host build of the
// same code (SA_RV's rvcode_host.so, firmware/rv/sa_rv_rt.c SA_RV_SIM), each
// command handed back here and run as a one-command list in sa_rv.c's
// scratch (after its argument block).
#include <dlfcn.h>
static void* sa_sim_rv_so;
static sa_transport_t* sa_sim_rv_tp;
static uint32_t sa_sim_rv_list, sa_sim_rv_err;
static iree_status_t sa_sim_rv_status;
static void sa_sim_rv_cmd(const uint64_t* w) {
  if (sa_sim_rv_err || !iree_status_is_ok(sa_sim_rv_status)) return;
  sa_transport_t* t = sa_sim_rv_tp;
  uint64_t* p = (uint64_t*)(t->mem + (sa_sim_rv_list - t->mem_phys));
  memcpy(p, w, 64);
  memset(p + 8, 0, 64);
  p[8] = 0x12;                                    // END 0x5A
  p[9] = 0x5A;
  __sync_synchronize();
  sa_completion_t done = {0};
  iree_status_t st = sa_sim_run(t, sa_sim_rv_list, &done);
  if (!iree_status_is_ok(st)) sa_sim_rv_status = st;
  else if (done.status) sa_sim_rv_err = done.status;
}
static uint32_t sa_sim_rv_load32(uint32_t a) {
  return *(volatile uint32_t*)(sa_sim_rv_tp->mem + (a - sa_sim_rv_tp->mem_phys));
}
static iree_status_t sa_sim_run_call(sa_transport_t* t, uint32_t code_phys, uint32_t args_phys, const char* symbol,
                                     sa_completion_t* out) {
  (void)code_phys;
  if (!sa_sim_rv_so) {
    const char* dir = getenv("SA_RV");
    char path[1024];
    snprintf(path, sizeof(path), "%s/rvcode_host.so", dir ? dir : ".");
    sa_sim_rv_so = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!sa_sim_rv_so) return iree_make_status(IREE_STATUS_NOT_FOUND, "sa sim: %s", dlerror());
    void (*init)(void (*)(const uint64_t*), uint32_t (*)(uint32_t)) =
        (void (*)(void (*)(const uint64_t*), uint32_t (*)(uint32_t)))dlsym(sa_sim_rv_so, "sa_rv_sim_init");
    if (!init) return iree_make_status(IREE_STATUS_NOT_FOUND, "sa sim: no sa_rv_sim_init in %s", path);
    init(sa_sim_rv_cmd, sa_sim_rv_load32);
  }
  void (*fn)(const uint32_t*, const uint32_t*) = (void (*)(const uint32_t*, const uint32_t*))dlsym(sa_sim_rv_so, symbol);
  if (!fn) return iree_make_status(IREE_STATUS_NOT_FOUND, "sa sim: no function %s in rvcode_host.so", symbol);
  sa_sim_rv_tp = t;
  sa_sim_rv_list = sa_rv_scratch_phys() + 256;     // (sa_rv.c's scratch: an argument block may sit in a table)
  sa_sim_rv_err = 0;
  sa_sim_rv_status = iree_ok_status();
  const uint32_t* args = (const uint32_t*)(t->mem + (args_phys - t->mem_phys));
  fn(args, args + 16);
  memset(out, 0, sizeof(*out));
  out->status = sa_sim_rv_err;
  out->end = 0x5A;
  iree_status_t st = sa_sim_rv_status;
  sa_sim_rv_status = iree_ok_status();
  return st;
}
#endif

// a call inside a schedule (RVS2 items: the code address, the symbol from sa_rv.c's map)
static uint32_t sa_sim_rvs_call(uint32_t code, uint32_t args);

// The decode schedule (firmware/include/rvsched.h) as rt_fw runs it on the
// board, its accesses on the shared window, its lists through RUN.
static sa_transport_t* sa_sim_rvs_t;
static iree_status_t sa_sim_rvs_error;
static uint32_t sa_sim_rvs_submit(uint32_t list) {
  sa_completion_t done = {0};
  iree_status_t st = sa_sim_run(sa_sim_rvs_t, list, &done);
  if (!iree_status_is_ok(st)) {
    if (iree_status_is_ok(sa_sim_rvs_error)) sa_sim_rvs_error = st; else iree_status_ignore(st);
    return 0xFFFFFFFFu;
  }
  return done.status ? done.status : (done.end != 0x5A ? 0xFFFFFFFEu : 0u);
}
#define RVS_RD(a) (*(volatile uint32_t*)(sa_sim_rvs_t->mem + ((uint32_t)(a) - sa_sim_rvs_t->mem_phys)))
#define RVS_WR(a, v) (RVS_RD(a) = (uint32_t)(v))
#define RVS_SUBMIT(l) sa_sim_rvs_submit(l)
#if defined(__x86_64__)
#define RVS_CALL(c, a) sa_sim_rvs_call(c, a)
#else
#define RVS_CALL(c, a) 0xFFFFFFFFu
#endif
#define RVS_CYCLES() 0u
#include "../../../firmware/include/rvsched.h"  // the decode schedule shared with rt_fw

#if defined(__x86_64__)
static uint32_t sa_sim_rvs_call(uint32_t code, uint32_t args) {
  sa_rv_host_fn_t r5 = sa_rv_host_kernel_at(code);   // (R5: a core kernel's host build; no unit commands)
  if (r5) {
    const uint32_t* a = (const uint32_t*)(sa_sim_rvs_t->mem + (args - sa_sim_rvs_t->mem_phys));
    r5(a, a + 16);
    return 0;
  }
  const char* sym = sa_rv_symbol_at(code);
  if (!sym) {
    if (iree_status_is_ok(sa_sim_rvs_error))
      sa_sim_rvs_error = iree_make_status(IREE_STATUS_NOT_FOUND, "sa sim: no function at %#x", code);
    return 0xFFFFFFFFu;
  }
  sa_completion_t done = {0};
  iree_status_t st = sa_sim_run_call(sa_sim_rvs_t, code, args, sym, &done);
  if (!iree_status_is_ok(st)) {
    if (iree_status_is_ok(sa_sim_rvs_error)) sa_sim_rvs_error = st; else iree_status_ignore(st);
    return 0xFFFFFFFFu;
  }
  return done.status;
}
#endif

static iree_status_t sa_sim_run_sched(sa_transport_t* t, uint32_t table_phys, double timeout_s, uint32_t* out_status) {
  (void)timeout_s;
  sa_sim_rvs_t = t;
  sa_sim_rvs_error = iree_ok_status();
  *out_status = rvs_run(table_phys);
  iree_status_t st = sa_sim_rvs_error;
  sa_sim_rvs_error = iree_ok_status();
  return st;
}


static void sa_sim_close(sa_transport_t* t) {
  sa_sim_t* s = (sa_sim_t*)t->impl;
  if (write(s->fd, "QUIT\n", 5) != 5) { /* the server may be gone */ }
  close(s->fd);
  munmap(t->mem, t->mem_size);
}

iree_status_t sa_transport_sim_open(iree_allocator_t host_allocator, sa_transport_t** out) {
  const char* path = getenv("SA_SIM_SOCKET");
  if (!path) path = "/tmp/sa_sim.sock";
  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);
  if (fd < 0 || connect(fd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
    if (fd >= 0) close(fd);
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "sa sim: cannot connect to %s (start compiler/sim/sa_sim_server.py)", path);
  }
  sa_transport_t* t = NULL;
  sa_sim_t* s = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*t) + sizeof(*s), (void**)&t));
  memset(t, 0, sizeof(*t) + sizeof(*s));
  s = (sa_sim_t*)(t + 1);
  s->fd = fd;
  char line[512], shm[400];
  unsigned d = 0, base = 0, size = 0;
  IREE_RETURN_IF_ERROR(sa_sim_readline(s, line, sizeof(line)));
  if (sscanf(line, "HELLO %u %399s %u %u", &d, shm, &base, &size) != 4) {
    return iree_make_status(IREE_STATUS_DATA_LOSS, "sa sim: unexpected greeting '%s'", line);
  }
  int mfd = open(shm, O_RDWR);
  void* mem = mfd >= 0 ? mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, mfd, 0) : MAP_FAILED;
  if (mfd >= 0) close(mfd);
  if (mem == MAP_FAILED) return iree_make_status(IREE_STATUS_UNAVAILABLE, "sa sim: cannot map %s", shm);
  t->name = "sim";
  t->mem = (uint8_t*)mem;
  t->mem_phys = base;
  t->mem_size = size;
  t->heap_offset = 0;
  t->d = d;
  t->fw_version = 6;      // (the schedule emulation is firmware/include/rvsched.h as it is)
  // the functional simulator has it all; SA_SIM_CAPS=<CAPS> stands in for another unit (tests)
  t->caps = (1u << 21) | (1u << 22) | (1u << 23) | (1u << 24) | (1u << 25);
  const char* caps = getenv("SA_SIM_CAPS");
  if (caps) t->caps = (uint32_t)strtoul(caps, NULL, 0);
  t->run = sa_sim_run;
  t->run_sched = sa_sim_run_sched;
#if defined(__x86_64__)
  t->run_call = sa_sim_run_call;
#endif
  t->close = sa_sim_close;
  t->impl = s;
  *out = t;
  return iree_ok_status();
}
