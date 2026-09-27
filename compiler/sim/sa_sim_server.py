#!/usr/bin/env python3
"""The simulator service: the accelerator for the sa HAL driver's `sim` transport
(docs/iree_compiler_plan.md §5.5).

It plays the part of the board: a shared-memory file is the DDR window that
the device sees (the driver maps the same file and allocates all device
buffers in it), and descriptor lists submitted through a Unix socket are run
by the functional simulator (llm/sa_funcsim.py) on that memory, bit-exact
with the hardware. The simulator keeps its state between lists, as the
hardware does (local memories, BASE / PARAM registers).

Protocol (one text line each way, one client at a time):
  server -> client on connect:  HELLO <D> <shm path> <physical base> <bytes>
  client -> server:             RUN <list physical address>
  server -> client:             DONE <status> <cycles> <descriptors> <end value>
      status 0 = success; otherwise the unit's extended status (xstatus), and
      <descriptors> counts the descriptors decoded, the failing one included (as rt_fw's completion
      record); cycles are 0 (no timing model)
  client -> server:             QUIT          (the server keeps listening)

    python3 compiler/sim/sa_sim_server.py [--d 8] [--mb 64] [--socket /tmp/sa_sim.sock]
"""
import argparse
import os
import socket
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "llm"))
from sa_funcsim import SaError, SaFuncSim  # noqa: E402


def xstatus(e):
    """The unit's extended status after a failed list, as rt_fw reports it
    (rtl/sysarray/sa_sched.v, sa_unit.v): [0] idle, [1] error, [11:8] code,
    [15:12] engine, [24] fetch busy (the fetch unit stops mid-list)."""
    return 1 << 24 | (e.engine & 0xF) << 12 | (e.code & 0xF) << 8 | 3

BASE = 0x10000000           # the simulated window's physical address (as tb_system / the funcsim tests)


def serve(args):
    size = args.mb << 20
    shm = args.shm or f"/dev/shm/sa_ddr_{os.getpid()}"
    with open(shm, "wb") as f:
        f.truncate(size)
    ddr = np.memmap(shm, np.uint8, "r+", shape=(size,))
    sim = SaFuncSim(args.d, BASE, size)
    sim.ddr = ddr                                       # the list and all buffers live in the shared file
    if os.path.exists(args.socket):
        os.remove(args.socket)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(args.socket)
    srv.listen(1)
    print(f"sa sim: D = {args.d}, {args.mb} MB at {BASE:#x} in {shm}, socket {args.socket}", flush=True)
    runs = 0
    try:
        while True:
            conn, _ = srv.accept()
            with conn, conn.makefile("rw") as io:
                io.write(f"HELLO {args.d} {shm} {BASE} {size}\n")
                io.flush()
                for line in io:
                    cmd = line.split()
                    if not cmd:
                        continue
                    if cmd[0] == "RUN":
                        addr = int(cmd[1], 0)
                        try:
                            n = sim.run_list(addr)
                            reply = f"DONE 0 0 {n} {sim.dl_status}"
                        except SaError as e:
                            status = xstatus(e)
                            reply = f"DONE {status} 0 {e.index + 1 if e.index is not None else sim.dl_exec} 0"
                            if args.verbose:
                                print(f"list {addr:#x}: {e}", flush=True)
                        runs += 1
                        if args.verbose:
                            print(f"run {runs}: list {addr:#x} -> {reply}", flush=True)
                        io.write(reply + "\n")
                        io.flush()
                    elif cmd[0] == "QUIT":
                        break
                    else:
                        io.write(f"ERROR unknown command {cmd[0]}\n")
                        io.flush()
            if args.once:
                break
    finally:
        srv.close()
        os.remove(args.socket)
        del ddr
        if not args.shm:
            os.remove(shm)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--mb", type=int, default=64, help="size of the simulated DDR window")
    ap.add_argument("--socket", default="/tmp/sa_sim.sock")
    ap.add_argument("--shm", help="shared-memory file (default /dev/shm/sa_ddr_<pid>)")
    ap.add_argument("--once", action="store_true", help="exit after the first client disconnects")
    ap.add_argument("-v", "--verbose", action="store_true")
    serve(ap.parse_args())


if __name__ == "__main__":
    main()
