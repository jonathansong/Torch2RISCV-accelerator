# Environment of the MLIR / IREE compiler work (docs/iree_compiler_plan.md).
# Source it:  source compiler/env.sh
#
# Everything large lives under build/ (git-ignored). By default the IREE
# source tree and the Python venv first set up for level L0 (llvm-cpu on the
# PYNQ-Z1's ARM; iree-sa/l0 on the pynq-z1 branch) are reused: the same IREE
# revision, so the runtime and the compiler built here match.

_SA_COMPILER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SA_REPO="$(cd "$_SA_COMPILER_DIR/.." && pwd)"
export SA_COMPILER="$_SA_COMPILER_DIR"

# IREE revision (the pip compiler of L0: iree-base-compiler 3.11.0)
export IREE_REV="${IREE_REV:-e4a3b0405d}"
export IREE_SRC="${IREE_SRC:-$SA_REPO/build/iree/src}"
export IREE_BUILD="${IREE_BUILD:-$SA_REPO/build/iree/build-compiler}"
export IREE_INSTALL="${IREE_INSTALL:-$SA_REPO/build/iree/install-compiler}"

# Python: torch, iree-turbine, iree-base-compiler / runtime (frontend, stage C0)
export SA_VENV="${SA_VENV:-$SA_REPO/build/iree/venv}"
export SA_PY="$SA_VENV/bin/python"

# Our plugins (compiled into iree-compile through IREE_CMAKE_PLUGIN_PATHS)
export SA_PLUGIN_PATHS="$SA_COMPILER/plugins/sa"

# Build parallelism (project convention: 4 jobs); links are memory hungry: 1 at a time
export SA_JOBS="${SA_JOBS:-4}"
export SA_LINK_JOBS="${SA_LINK_JOBS:-1}"

# The iree-compile built here, when present, comes first
if [ -x "$IREE_BUILD/tools/iree-compile" ]; then
  export PATH="$IREE_BUILD/tools:$PATH"
fi
