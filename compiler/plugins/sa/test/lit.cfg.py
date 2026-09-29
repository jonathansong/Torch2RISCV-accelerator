# lit configuration of the sa plugin's pass tests (docs/iree_compiler_plan.md §8.6):
#   $IREE_BUILD/llvm-project/bin/llvm-lit -v compiler/plugins/sa/test
# (compiler/scripts/run_tests.sh golden runs it). Tools come from $IREE_BUILD.
import os

import lit.formats

config.name = "sa-plugin"
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
build = os.environ.get("IREE_BUILD", os.path.join(os.path.dirname(__file__), "../../../../build/iree/build-compiler"))
config.substitutions.append(("iree-opt", os.path.join(build, "tools", "iree-opt")))
config.substitutions.append(("iree-compile", os.path.join(build, "tools", "iree-compile")))
config.substitutions.append(("FileCheck", os.path.join(build, "llvm-project", "bin", "FileCheck")))
