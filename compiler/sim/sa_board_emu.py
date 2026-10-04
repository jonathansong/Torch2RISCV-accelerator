#!/usr/bin/env python3
"""Host emulation of the board path of the sa HAL driver (compiler/runtime/sa/
sa_transport_board.c): rt_fw's submission ring served by the functional
simulator, so the board transport's ring protocol is tested before the board.

A sparse file stands in for /dev/mem (physical address = file offset): the
DDR window at --base and the BRAM mailbox at 0x40011F00. Like the PYNQ
launcher (compiler/runtime/test/board_launcher.py) it puts the ring at the
start of the window (completion records at +0x2000) and writes RING_BASE /
RING_SIZE / CPL_BASE; like rt_fw (firmware/rt/rt_fw.c) it then sets FW_STATE =
RT_READY, polls RING_TAIL, runs each RUN_LIST entry (BASE0..3 from the entry,
optional parameter block) and writes the completion record and RING_HEAD.

    python3 compiler/sim/sa_board_emu.py --d 8 --file /dev/shm/sa_devmem -- <program> <args>
runs the program with SA_TRANSPORT=board, SA_BOARD_DEVMEM and SA_BOARD_* set,
serving the ring until it exits; returns its exit code.
"""
import argparse
import os
import subprocess
import sys
import threading

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "llm"))
sys.path.insert(0, os.path.join(REPO, "driver"))
from pynq_matmul import (MBOX, BRAM_ARM_BASE, MBOX_CPL_BASE, MBOX_FW_STATE, MBOX_RING_BASE,  # noqa: E402
                         MBOX_RING_HEAD, MBOX_RING_SIZE, MBOX_RING_TAIL, RT_READY, RT_RUN_LIST)
from sa_funcsim import SaError, SaFuncSim  # noqa: E402


def xstatus(e):
    """The unit's extended status after a failed list, as rt_fw reports it
    (rtl/sysarray/sa_sched.v, sa_unit.v): [0] idle, [1] error, [11:8] code,
    [15:12] engine, [24] fetch busy (the fetch unit stops mid-list)."""
    return 1 << 24 | (e.engine & 0xF) << 12 | (e.code & 0xF) << 8 | 3

MBOX_PHYS = BRAM_ARM_BASE + MBOX
CPL_OFFSET = 0x2000


def serve(sim, dev, base, ring_entries, stop, log):
    mb = dev[MBOX_PHYS:MBOX_PHYS + 0x100].view("<u4")
    head = 0
    while not stop.is_set():
        if int(mb[MBOX_RING_TAIL // 4]) == head:
            continue
        slot = head & (ring_entries - 1)
        e = dev[base + 64 * slot:base + 64 * slot + 64].view("<u8")
        c = dev[base + CPL_OFFSET + 32 * slot:base + CPL_OFFSET + 32 * slot + 32].view("<u4")
        w0 = int(e[0])
        typ, seq = w0 & 0xFF, w0 >> 32
        status = exe = end = 0
        if typ == RT_RUN_LIST:
            bases = [int(e[3 + i]) & 0xFFFFFFFF for i in range(4)]
            pb = int(e[7]) & 0xFFFFFFFF
            params = [int(v) for v in dev[pb:pb + 32].view("<u4")] if pb else None
            try:
                exe = sim.run_list(int(e[1]) & 0xFFFFFFFF, int(e[2]) & 0xFFFFFFFF, bases, params)
                end = sim.dl_status
            except SaError as err:
                status = xstatus(err)
                exe = err.index + 1 if err.index is not None else sim.dl_exec
        c[1:8] = [status, 0, exe, end, 0, 0, 0]
        c[0] = seq
        head += 1
        mb[MBOX_RING_HEAD // 4] = head
        log.append((seq, typ, status, exe, end))


def main():
    global MBOX_PHYS                     # (serve() reads it)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--mb", type=int, default=64)
    ap.add_argument("--base", type=lambda s: int(s, 0), default=0x10000000)
    ap.add_argument("--ring", type=int, default=16)
    ap.add_argument("--file", default=f"/dev/shm/sa_devmem_{os.getpid()}")
    ap.add_argument("--mbox", type=lambda s: int(s, 0), default=MBOX_PHYS,
                    help="the mailbox's physical address (KV260: 0xA0011F00)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    size = args.mb << 20
    MBOX_PHYS = args.mbox
    span = MBOX_PHYS + 0x100
    with open(args.file, "wb") as f:
        f.truncate(span)
    dev = np.memmap(args.file, np.uint8, "r+", shape=(span,))
    sim = SaFuncSim(args.d, args.base, size)
    sim.ddr = dev[args.base:args.base + size]
    mb = dev[MBOX_PHYS:MBOX_PHYS + 0x100].view("<u4")
    mb[MBOX_RING_BASE // 4] = args.base
    mb[MBOX_RING_SIZE // 4] = args.ring
    mb[MBOX_CPL_BASE // 4] = args.base + CPL_OFFSET
    mb[MBOX_FW_STATE // 4] = RT_READY
    stop, log = threading.Event(), []
    t = threading.Thread(target=serve, args=(sim, dev, args.base, args.ring, stop, log), daemon=True)
    t.start()
    env = dict(os.environ, SA_TRANSPORT="board", SA_BOARD_DEVMEM=args.file,
               SA_BOARD_MEM=f"{args.base:#x}:{size:#x}", SA_BOARD_MBOX=f"{MBOX_PHYS:#x}",
               SA_BOARD_RING=str(args.ring), SA_BOARD_D=str(args.d))
    try:
        rc = subprocess.call(cmd, env=env)
    finally:
        stop.set()
        t.join()
        del dev
        os.remove(args.file)
    print(f"sa board emu: {len(log)} ring entries served, RING_HEAD {len(log)}; statuses "
          f"{sorted(set(s for _, _, s, _, _ in log))}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
