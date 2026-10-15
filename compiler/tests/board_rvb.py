#!/usr/bin/env python3
"""RISC-V backend test on the board (docs/riscv_backend_design.md §5, R1 / R3):
each case of cases/ (compiler/tests/rv_dispatch.py: one compiled decode
dispatch, its DDR window before and after the functional simulator's run)
run by the PicoRV32 firmware firmware/rvb twice, as in the testbench's
RVB_TEST: its descriptor list (mat_submit) and the same commands over PCPI
(rvb_gen), the generated code in the program BRAM (fw_bram.bin) and in DDR
(fw_ddr.bin + code.bin, fetched uncached over HP0). The window is a buffer
wherever pynq allocates it: the list is rebuilt there (rvb_list.json: the
SETREG values, BASE ones relative to the window), the code takes the window's
address from the mailbox. After each run the bindings are compared with the
functional simulator's bytes; in between they are restored.

    python3 board_rvb.py [--cases smol_7,smol_16] [--modes bram,ddr] [--timeout 5]
    (or python3 board_run.py rvb: the results under results/<commit>/<bundle>/<overlay>/rvb/)
Writes rvb_board.json; prints one line per case and mode: cycles (rdcycle, the
unit's clock) of the list and of PCPI, their ratio, OK / MISMATCH.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pynq import allocate  # noqa: E402
from pynq_matmul import DescList, MatmulOverlay  # noqa: E402

RV = 0x1FD8                 # firmware/rvb/rvb_fw.c RV_MBOX: program BRAM offset
RV_PHASE, RV_GO, RV_CYC_DESC, RV_CYC_PCPI, RV_ST_DESC, RV_ST_PCPI, RV_MAGIC, RV_BASE, RV_CODE = range(9)
CODE_SPACE = 65536          # after the window: the DDR copy of rvb_gen


def mb_w(mm, i, v):
    mm.bram.write(RV + 4 * i, int(v) & 0xFFFFFFFF)


def mb_r(mm, i):
    return mm.bram.read(RV + 4 * i)


def wait_phase(mm, phase, timeout):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if mb_r(mm, RV_PHASE) == phase:
            return True
    return False


def compare(buf, exp, ranges):
    bad = 0
    for o, n in ranges:
        got = np.asarray(buf[o:o + n])
        bad += int(np.count_nonzero(got != exp[o:o + n]))
    return bad


def run_case(mm, case, mode, timeout):
    d = os.path.join(HERE, "cases", case)
    meta = json.load(open(os.path.join(d, "rvb_list.json")))
    img = np.fromfile(os.path.join(d, "rvb_ddr.bin"), np.uint8)
    exp = np.fromfile(os.path.join(d, "rvb_expect.bin"), np.uint8)
    ranges = [tuple(map(int, line.split())) for line in open(os.path.join(d, "rvb_cmp.txt")) if line.strip()]
    win = meta["window"]
    buf = allocate(shape=(win + CODE_SPACE,), dtype=np.uint8)
    try:
        P = buf.physical_address
        buf[:len(img)] = img
        dl = DescList()                                  # the driver's list at the buffer's address
        regs = [(i, P + v if rel else v) for i, v, rel in meta["regs"]]
        for k in range(0, len(regs), 3):
            dl.setreg(*regs[k:k + 3])
        for c in meta["calls"]:
            dl.call(P + c)
        dl.end(meta["end"])
        lst = dl.array().view(np.uint8).ravel()
        lo = meta["list_off"]
        buf[lo:lo + len(lst)] = lst
        if mode == "ddr":
            code = np.fromfile(os.path.join(d, "code.bin"), np.uint8)
            buf[win:win + len(code)] = code
        buf.flush()
        mm.load_firmware(os.path.join(d, f"fw_{mode}.bin"))      # (holds the core in reset)
        for i in range(9):
            mb_w(mm, i, 0)
        mb_w(mm, RV_BASE, P)
        mb_w(mm, RV_CODE, P + win)
        mm.reset.write(0)
        res = {"case": case, "dispatch": meta["name"], "t": meta["t"], "mode": mode}
        if not wait_phase(mm, 1, timeout):
            res["error"] = f"phase 1 not reached (phase {mb_r(mm, RV_PHASE)}, magic {mb_r(mm, RV_MAGIC):#x})"
            return res
        buf.invalidate()
        res["bad_desc"] = compare(buf, exp, ranges)
        for o, n in ranges:                              # the bindings as before the run
            buf[o:o + n] = img[o:o + n]
        buf.flush()
        mb_w(mm, RV_GO, 2)
        if not wait_phase(mm, 2, timeout):
            res["error"] = f"phase 2 not reached (phase {mb_r(mm, RV_PHASE)})"
            return res
        buf.invalidate()
        res["bad_pcpi"] = compare(buf, exp, ranges)
        res.update(cyc_desc=mb_r(mm, RV_CYC_DESC), cyc_pcpi=mb_r(mm, RV_CYC_PCPI),
                   st_desc=mb_r(mm, RV_ST_DESC), st_pcpi=mb_r(mm, RV_ST_PCPI))
        return res
    finally:
        mm.reset.write(1)
        buf.freebuffer()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", help="comma-separated (default: all of cases/)")
    ap.add_argument("--modes", default="bram,ddr")
    ap.add_argument("--timeout", type=float, default=5.0)
    args = ap.parse_args()
    cases = args.cases.split(",") if args.cases else sorted(os.listdir(os.path.join(HERE, "cases")))
    mm = MatmulOverlay(os.path.join(HERE, "picorv32.bit"),
                       os.path.join(HERE, "cases", cases[0], "fw_bram.bin"))
    print(f"overlay: D = {mm.d}, {mm.nports} ports, RISC-V {mm.riscv_hz / 1e6:.0f} MHz")
    out, fails = [], 0
    for case in cases:
        for mode in args.modes.split(","):
            r = run_case(mm, case, mode, args.timeout)
            out.append(r)
            if "error" in r:
                fails += 1
                print(f"{case:8s} {mode:4s} ERROR {r['error']}")
                continue
            ok = r["bad_desc"] == 0 and r["bad_pcpi"] == 0 and not (r["st_desc"] & 2) and not (r["st_pcpi"] & 2)
            fails += not ok
            print(f"{case:8s} {mode:4s} {r['dispatch'][:48]:48s} list {r['cyc_desc']:8d}  PCPI {r['cyc_pcpi']:8d}  "
                  f"x{r['cyc_pcpi'] / max(r['cyc_desc'], 1):.2f}  {'OK' if ok else 'MISMATCH'}"
                  + ("" if ok else f" (bytes {r['bad_desc']} / {r['bad_pcpi']}, status {r['st_desc']:#x} / {r['st_pcpi']:#x})"))
    json.dump(out, open(os.path.join(HERE, "rvb_board.json"), "w"), indent=1)
    print(f"board_rvb {'PASS' if not fails else 'FAIL'}: {len(out)} runs, {fails} failing")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
