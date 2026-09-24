# Double-buffered accelerator (docs/double_buffer_design.md).
#   make sim TB=tb_sa_ex      run one testbench from sim/ in Vivado xsim
#   make sim TB=tb_sa_dma GEN=NP=3   ... with a top-level generic
#   make test                 run all unit testbenches
#   make synth                out-of-context synth/place/route (CLOCK_NS, SYNTH_GEN)
#   make clean

VIVADO_SETTINGS ?= /home/jon/Projects/Vivado/Vivado/2024.1/settings64.sh
PYTHON          ?= python3
TB              ?= tb_sa_ex
GEN             ?=

RTL := $(abspath $(wildcard *.v))
SIM := build/$(TB)$(subst =,,$(GEN))

VECTORS := $(abspath ../matmul/sim/gen_vectors.py)

sim:
	mkdir -p $(SIM)
	$(if $(filter tb_sa_unit,$(TB)),$(PYTHON) $(VECTORS) --out $(SIM) >/dev/null)
	cd $(SIM) && bash -c 'source $(VIVADO_SETTINGS) >/dev/null && \
	  xvlog -i $(CURDIR) $(RTL) $(abspath sim/$(TB).v) >xvlog.out 2>&1 || { grep -E "ERROR|WARNING: \[VRFC" xvlog.out; exit 1; }; \
	  xelab -debug off $(if $(GEN),-generic_top "$(GEN)") $(TB) -s tb >xelab.out 2>&1 || { grep ERROR xelab.out; exit 1; }; \
	  xsim tb -R' | grep -E "^(TB|ERROR|FATAL)"

CLOCK_NS ?= 20.0
SYNTH_GEN ?= D=8 NPORTS=1

synth:
	mkdir -p build/synth
	cd build/synth && bash -c 'source $(VIVADO_SETTINGS) >/dev/null && vivado -mode batch -notrace -nojournal \
	  -log vivado.log -source $(abspath synth_ooc.tcl) -tclargs $(CLOCK_NS) $(CURDIR) $(SYNTH_GEN)' \
	  | grep -E "^(RESULT|ERROR|CRITICAL)"

test:
	@$(MAKE) -s sim TB=tb_sa_ex
	@$(MAKE) -s sim TB=tb_sa_dma GEN=NP=1
	@$(MAKE) -s sim TB=tb_sa_dma GEN=NP=3
	@$(MAKE) -s sim TB=tb_sa_unit
	@$(MAKE) -s sim TB=tb_sa_ve

clean:
	rm -rf build

.PHONY: sim test clean
