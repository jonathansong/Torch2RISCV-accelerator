# iree-sa/l0: IREE on the PYNQ-Z1 ARM (toolchain check)

This is level L0, §4.4 of [docs/llm_inference_plan.md](../../docs/llm_inference_plan.md).
It checks that a torch model can be compiled by IREE for the board's ARM
and run there by the IREE runtime. The full path is: torch → iree-turbine →
iree-compile (llvm-cpu, armv7) → IREE runtime on the Cortex-A9. In the
real system the ARM runs the dispatches that the accelerator does not take,
and it runs the runtime and HAL driver of L6-IREE.

| File | What |
|---|---|
| `build_runtime_armv7.sh` | One command does everything. It fetches the IREE source at the pip compiler's revision, with only the runtime submodules. It builds the host tools, then cross-builds `iree-run-module`, `iree-benchmark-module` and `iree-cpuinfo` for armv7, statically linked. It compiles the test models and stages `build/deploy_iree_l0/`. |
| `toolchain-armv7hf.cmake` | The cross build uses clang-18 and the Ubuntu armhf cross sysroot, with `-march=armv7-a -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard -static`. Linking is static because the host sysroot has glibc 2.39 and the board has 2.35. |
| `export_and_compile.py` | Exports and compiles the test model (RMSNorm, a 2-layer SiLU MLP, softmax) for the host and for armv7. It checks the host result against torch and checks the armv7 executables for imports. |
| `op_tests.py`, `run_op_tests.sh` | 15 single-op modules, used to find where armv7 codegen goes wrong. |
| `armv7_libm_shim.c`, `embedded_ld.sh` | Fixes for two armv7 problems, described below. |
| `run_iree_l0.sh` | Board script. |

```sh
iree-sa/l0/build_runtime_armv7.sh                     # host; needs build/iree/venv (iree-base-compiler,
                                                      # iree-base-runtime, iree-turbine, torch CPU)
scp -r build/deploy_iree_l0 xilinx@<board>:/home/xilinx/iree_l0
cd /home/xilinx/iree_l0 && bash run_iree_l0.sh        # board, no sudo
```

## Result (board, 2026-09-25, IREE 3.11.0 at e4a3b0405d)

**IREE L0 PASS.** The test model matches torch with both the `local-sync`
and `local-task` drivers. All 15 single-op tests pass: add, mul, linear,
exp, amax, sum, maximum, relu, rsqrt, div, sigmoid, silu, softmax, x − max
and exp(x − max). One call of the test model (4×64 input) takes 0.88 ms with
`local-task` on 2 threads.

## Two armv7 problems and their fixes

IREE links llvm-cpu code into a platform-independent embedded ELF. The
embedded loader rejects any imported symbol. On ARMv7 without VFPv4, two
things go wrong.

1. **`fmaxf`/`fminf` are missing.** Cortex-A9 (VFPv3 + NEON) has no IEEE
   maxNum/minNum instruction. LLVM therefore calls libm for max and min,
   which softmax and ReLU use, and the embedded ELF has no libm.
   `armv7_libm_shim.c` provides them with IEEE NaN and signed-zero
   semantics.
2. **The bundled `fmaf` is an empty stub.** The Cortex-A9 has no FMA. For
   every `llvm.fma` LLVM calls `fmaf`, for example 36 times in IREE's
   polynomial `exp`. The `fmaf` in IREE's bundled musl bitcode has the body
   `unreachable` (a LOCAL symbol of size 0). The call runs into unrelated
   code and `exp` returned a constant 4.
   - The shim has a correctly rounded software `fmaf`. It uses musl's
     generic algorithm without the fenv parts. It is bit-exact against
     glibc on 4·10⁷ random inputs.
   - `embedded_ld.sh` runs `llvm-objcopy` to make the bundled `fmaf` a weak
     global, so the shim's `fmaf` is linked instead.

   This is probably worth reporting to IREE.

All shim symbols are `hidden`, so calls bind at link time with no PLT. The
shim also defines the ARM EHABI personality routines
(`__aeabi_unwind_cpp_pr0/1/2`) as trapping stubs, because the `.ARM.exidx`
tables reference them.

`export_and_compile.py` checks every linked armv7 executable on the host.
It fails if the executable imports any symbol, or if `fmaf`, `fmaxf` or
`fminf` is still empty.
