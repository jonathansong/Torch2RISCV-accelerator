#!/usr/bin/env python3
"""C1 on the board (docs/iree_compiler_plan.md §5.6): start the accelerator
for the sa HAL driver's board transport and run a program against it.

The PYNQ side does what the IREE runtime cannot do by itself: load the
overlay, allocate a physically contiguous window (CMA) for all device memory,
put rt_fw's submission ring at its start (completion records at +0x2000;
compiler/runtime/sa/sa_transport_board.c) and release the RISC-V running
rt_fw. The program (the static aarch64 sa_hal_test, or iree-run-module with
--device=sa) then maps the window and the mailbox through /dev/mem and talks
to rt_fw directly; this script only waits for it.

Files in one directory on the board: picorv32.bit / .hwh (L2 build),
rt_fw.bin, pynq_matmul.py, this script, sa_hal_test, test_d8/ (from
compiler/runtime/tools/make_test_exec.py --d 8).

    sudo bash -c 'source /etc/profile.d/pynq_venv.sh && source /etc/profile.d/xrt_setup.sh && \\
        cd /home/xilinx/c1 && python3 board_launcher.py -- ./sa_hal_test test_d8'
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pynq_matmul import (MBOX, MBOX_CPL_BASE, MBOX_FW_STATE, MBOX_HEARTBEAT,  # noqa: E402
                         MBOX_RING_BASE, MBOX_RING_HEAD, MBOX_RING_SIZE, MBOX_WORDS, RT_READY, MatmulOverlay,
                         allocate)

CPL_OFFSET = 0x2000                      # sa_transport_board.c
UDMABUF = os.environ.get("SA_UDMABUF", "/dev/udmabuf0")


class UdmabufWindow:
    """The device window from the u-dma-buf module (KV260: arm64 kernels with
    CONFIG_STRICT_DEVMEM refuse /dev/mem on RAM, so the program cannot map a
    PYNQ buffer). The module allocated a contiguous buffer at load
    (`insmod u-dma-buf.ko udmabuf0=<bytes>`); opened with O_SYNC its mapping is
    non-cached, as the program's. The first `mb` MB serve as the window; the
    program maps the same device (SA_BOARD_MEM_DEV)."""

    def __init__(self, dev, mb):
        import mmap
        name = os.path.basename(dev)
        sysfs = next((d for d in (f"/sys/class/u-dma-buf/{name}", f"/sys/class/udmabuf/{name}") if os.path.isdir(d)), None)
        if sysfs is None:
            raise RuntimeError(f"{dev}: no sysfs entry (is the u-dma-buf module loaded?)")
        self.physical_address = int(open(os.path.join(sysfs, "phys_addr")).read().strip(), 0)
        size = int(open(os.path.join(sysfs, "size")).read().strip(), 0)
        self.nbytes = mb << 20
        if self.nbytes > size:
            raise RuntimeError(f"{dev}: {size >> 20} MB, the window needs {mb} MB (reload u-dma-buf larger)")
        self.dev = dev
        # sync_mode 2: an O_SYNC mapping is write-combining (arm64: Normal
        # non-cacheable). The default 1 maps it pgprot_noncached, which on arm64
        # is Device memory: memcpy / memset would fault on unaligned accesses.
        mode = os.path.join(sysfs, "sync_mode")
        if os.path.exists(mode) and open(mode).read().strip() != "2":
            with open(mode, "w") as f:
                f.write("2")
        fd = os.open(dev, os.O_RDWR | os.O_SYNC)
        try:
            self._map = mmap.mmap(fd, self.nbytes, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        finally:
            os.close(fd)
        self.view = np.frombuffer(self._map, dtype=np.uint8)

    def __setitem__(self, key, value):
        self.view[key] = value

    def flush(self):                     # (non-cached: nothing to write back)
        pass

    def freebuffer(self):
        self.view = None
        self._map.close()


def alloc_window(mb):
    """The device window: u-dma-buf when its device exists (KV260), else a PYNQ
    buffer in CMA (PYNQ-Z1)."""
    if os.path.exists(UDMABUF):
        return UdmabufWindow(UDMABUF, mb)
    return alloc_pynq_window(mb)


def alloc_pynq_window(mb):
    """The device window (CMA). A large window can fail on free but fragmented
    CMA (loading the overlay leaves page cache in it): then drop the page cache,
    compact memory (root) and retry. SmolLM2 (168 MB) needs cma=320M on the
    PYNQ-Z1 (uEnv.txt bootargs): with 256M the overlay load leaves no 168 MB block."""
    for attempt in range(4):
        try:
            return allocate(shape=(mb << 20,), dtype=np.uint8)
        except RuntimeError as e:
            if attempt == 3:
                raise RuntimeError(f"{mb} MB window: {e} (CMA: {cma_info()}; is cma= large enough, "
                                   "Jupyter stopped?)") from None
            print(f"launcher: {mb} MB window: {e}; dropping caches, compacting memory, retrying", flush=True)
            os.sync()
            for f, v in (("/proc/sys/vm/drop_caches", "3"), ("/proc/sys/vm/compact_memory", "1")):
                if os.path.exists(f):            # compact_memory: only with CONFIG_COMPACTION
                    with open(f, "w") as fh:
                        fh.write(v)
            time.sleep(1)


def cma_info():
    try:
        return ", ".join(l.split(":")[0] + " " + l.split()[1] + " kB" for l in open("/proc/meminfo") if l.startswith("Cma"))
    except OSError:
        return "unknown"


def start(bit, fw, mb=16, ring=16):
    """Overlay + window + ring + rt_fw; returns (mm, buf, env for the program)."""
    mm = MatmulOverlay(bit, fw)
    buf = alloc_window(mb)
    buf[:] = 0
    buf.flush()                          # no dirty lines left: the program maps the window non-cacheable
    phys = buf.physical_address
    bram = mm.bram
    for w in range(MBOX_WORDS):
        bram.write(MBOX + 4 * w, 0)
    bram.write(MBOX + MBOX_RING_BASE, phys)
    bram.write(MBOX + MBOX_RING_SIZE, ring)
    bram.write(MBOX + MBOX_CPL_BASE, phys + CPL_OFFSET)
    mm.reset.write(0)
    t0 = time.perf_counter()
    while mm._mbox(MBOX_FW_STATE) != RT_READY:
        if time.perf_counter() - t0 > 5:
            raise RuntimeError(f"rt_fw not ready (FW_STATE {mm._mbox(MBOX_FW_STATE):#x})")
    kind = f"u-dma-buf {buf.dev}" if isinstance(buf, UdmabufWindow) else "PYNQ buffer"
    print(f"launcher: {mm.info.board} overlay D = {mm.d}, {mm.riscv_hz / 1e6:.1f} MHz, rt_fw ready, "
          f"window {mb} MB at {phys:#x} ({kind}), ring {ring}", flush=True)
    if isinstance(buf, UdmabufWindow):   # the program maps the window from the same device
        os.environ["SA_BOARD_MEM_DEV"] = buf.dev
    else:
        os.environ.pop("SA_BOARD_MEM_DEV", None)
    env = dict(os.environ, SA_TRANSPORT="board", SA_DEVICE_HZ=f"{mm.riscv_hz:.0f}", SA_BOARD_MEM=f"{phys:#x}:{mb << 20:#x}",
               SA_BOARD_MBOX=f"{mm.bram_base + MBOX:#x}", SA_BOARD_RING=str(ring), SA_BOARD_D=str(mm.d))
    return mm, buf, env


def stop(mm, buf):
    """Holds the RISC-V in reset and frees the window; returns (RING_HEAD, heartbeat)."""
    head, beat = mm._mbox(MBOX_RING_HEAD), mm._mbox(MBOX_HEARTBEAT)
    mm.reset.write(1)
    buf.freebuffer()
    return head, beat


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bit", default=os.path.join(HERE, "picorv32.bit"))
    ap.add_argument("--fw", default=os.path.join(HERE, "rt_fw.bin"))
    ap.add_argument("--mb", type=int, default=16, help="device memory window (MB)")
    ap.add_argument("--ring", type=int, default=16, help="ring entries (power of 2, at most 128)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd:
        ap.error("no program given")
    mm, buf, env = start(args.bit, args.fw, args.mb, args.ring)
    rc = 1
    try:
        rc = subprocess.call(cmd, env=env)
    finally:
        head, beat = stop(mm, buf)
    print(f"launcher: program exit {rc}; rt_fw completed {head} entries (heartbeat {beat})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
