# 8x8x8 int8 matmul unit (Phase 2).
#   make sim     NumPy golden vectors + Vivado xsim testbench
#   make synth   out-of-context synth + place + route on xc7z020 (reports in build/synth)
#   make clean

VIVADO_SETTINGS ?= /home/jon/Projects/Vivado/Vivado/2024.1/settings64.sh
PYTHON          ?= python3
CLOCK_NS        ?= 10.0

RTL   := $(abspath systolic_array.v matmul_unit.v)
TB    := $(abspath sim/tb_matmul_unit.v)
SIM   := build/sim
SYNTH := build/synth

VIVADO_ENV = source $(VIVADO_SETTINGS) >/dev/null &&

all: sim

$(SIM)/n_cases.vh: sim/gen_vectors.py
	mkdir -p $(SIM)
	$(PYTHON) sim/gen_vectors.py --out $(SIM)

sim: $(SIM)/n_cases.vh
	cd $(SIM) && bash -c '$(VIVADO_ENV) \
	  xvlog -i . $(RTL) $(TB) >xvlog.out 2>&1 || { cat xvlog.out; exit 1; }; \
	  xelab -debug off tb_matmul_unit -s tb >xelab.out 2>&1 || { cat xelab.out; exit 1; }; \
	  xsim tb -R' | grep -E "^(TB|ERROR|FATAL)"

synth:
	mkdir -p $(SYNTH)
	cd $(SYNTH) && bash -c '$(VIVADO_ENV) vivado -mode batch -notrace -nojournal -log vivado.log \
	  -source $(abspath synth_ooc.tcl) -tclargs $(CLOCK_NS) $(RTL)' | grep -E "^(RESULT|ERROR|CRITICAL)"

clean:
	rm -rf build

.PHONY: all sim synth clean
