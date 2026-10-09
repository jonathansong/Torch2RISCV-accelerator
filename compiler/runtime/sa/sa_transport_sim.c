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
  // the functional simulator has it all; SA_SIM_CAPS=<CAPS> stands in for another unit (tests)
  t->caps = (1u << 21) | (1u << 22) | (1u << 23) | (1u << 24) | (1u << 25);
  const char* caps = getenv("SA_SIM_CAPS");
  if (caps) t->caps = (uint32_t)strtoul(caps, NULL, 0);
  t->run = sa_sim_run;
  t->close = sa_sim_close;
  t->impl = s;
  *out = t;
  return iree_ok_status();
}
