#!/usr/bin/env bash
# Adds the compiler's submodules (llvm-project, torch-mlir) to the IREE source
# tree of level L0 (which has the runtime submodules only). Shallow, at the
# revisions IREE pins. stablehlo is not compiled (IREE_INPUT_STABLEHLO=OFF),
# but IREE's configure step requires every submodule to be initialized.
set -euo pipefail
source "$(dirname "$0")/../env.sh"

if [ ! -d "$IREE_SRC/.git" ]; then
  echo "cloning IREE $IREE_REV into $IREE_SRC"
  mkdir -p "$IREE_SRC"
  git -C "$IREE_SRC" init -q
  git -C "$IREE_SRC" remote add origin https://github.com/iree-org/iree.git
  git -C "$IREE_SRC" fetch -q --depth 1 origin "$IREE_REV"
  git -C "$IREE_SRC" checkout -q FETCH_HEAD
  (cd "$IREE_SRC" && bash build_tools/scripts/git/update_runtime_submodules.sh)
fi
have=$(git -C "$IREE_SRC" rev-parse --short=10 HEAD)
if [ "$have" != "${IREE_REV:0:10}" ]; then
  echo "error: $IREE_SRC is at $have, expected $IREE_REV" >&2
  exit 1
fi

avail=$(df -BG --output=avail "$IREE_SRC" | tail -1 | tr -dc 0-9)
echo "IREE $have in $IREE_SRC; free disk ${avail} GB"
if [ "$avail" -lt 6 ]; then
  echo "error: need about 5 GB for the llvm-project and torch-mlir sources" >&2
  exit 1
fi

for sm in third_party/llvm-project third_party/torch-mlir third_party/stablehlo; do
  echo "== $sm"
  git -C "$IREE_SRC" submodule update --init --depth 1 --single-branch "$sm"
done
git -C "$IREE_SRC" submodule status third_party/llvm-project third_party/torch-mlir third_party/stablehlo
du -sh "$IREE_SRC"
