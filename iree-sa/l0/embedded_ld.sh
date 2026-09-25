#!/usr/bin/env bash
# iree-compile --iree-llvmcpu-embedded-linker-path=<this script> for armv7.
# Runs IREE's own lld with armv7_libm_shim.o added (see there), after making
# the empty `fmaf` from IREE's bundled musl bitcode (a LOCAL symbol whose body
# is `unreachable`) a weak global in each input object, so the shim's
# correctly rounded fmaf is the one that gets linked.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
LLD=${IREE_LLD:-$(dirname "$(command -v iree-compile)")/../lib/python3*/site-packages/iree/compiler/_mlir_libs/iree-lld}
OBJCOPY=${OBJCOPY:-llvm-objcopy-18}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
args=()
for a in "$@"; do
    if [[ $a == *.o && -f $a ]]; then
        o=$TMP/$(basename "$a")
        "$OBJCOPY" --globalize-symbol=fmaf --weaken-symbol=fmaf "$a" "$o"
        args+=("$o")
    else
        args+=("$a")
    fi
done
$LLD "${args[@]}" "$HERE/build/armv7_libm_shim.o"
