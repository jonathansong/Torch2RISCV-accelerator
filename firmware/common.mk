# Shared build + system-simulation rules for the PicoRV32 firmware.
# A firmware directory sets FW (output name) and SRCS, then includes this:
#   make        -> $(FW).bin  (load into the program BRAM; see driver/pynq_matmul.py)
#   make sim    -> sim/tb_system.v: the firmware on picorv32_axi driving the
#                  (SIM_D=16: the D = 16 unit)
#                  real sa_unit (CSR + PCPI) against a DDR model
#                  (SIM_DEFINES: GEMM_TEST for the GEMM firmware)
# clang/lld, rv32imc (matches the overlay's PicoRV32 configuration).

FW_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT    := $(abspath $(FW_ROOT)/..)

CC      := clang
OBJCOPY := llvm-objcopy
CFLAGS  := --target=riscv32-unknown-elf -march=rv32imc -mabi=ilp32 \
           -O2 -ffreestanding -fno-builtin -nostdlib -Wall -Wextra \
           -I$(FW_ROOT)/include
LDFLAGS := -fuse-ld=lld -T $(FW_ROOT)/common/link.ld -Wl,--gc-sections

VIVADO_SETTINGS ?= /home/jon/Projects/Vivado/Vivado/2024.1/settings64.sh
PYTHON          ?= python3

SA  := $(ROOT)/rtl/sysarray
RTL := $(ROOT)/RISCV-on-PYNQ-Z1/picorv32/picorv32.v $(wildcard $(SA)/*.v)
SIM_DEFINES ?=
# array size of the simulated unit (8 or 16)
SIM_D       ?= 8
SIM := build/sim$(SIM_D)

all: $(FW).bin

$(FW).elf: $(FW_ROOT)/common/start.S $(SRCS) $(wildcard $(FW_ROOT)/include/*.h) $(FW_ROOT)/common/link.ld
	$(CC) $(CFLAGS) $(LDFLAGS) $(FW_ROOT)/common/start.S $(SRCS) -o $@

$(FW).bin: $(FW).elf
	$(OBJCOPY) -O binary $< $@

$(SIM)/fw.hex: $(FW).bin
	mkdir -p $(SIM)
	$(PYTHON) -c "import struct; d=open('$<','rb').read(); d+=b'\0'*(-len(d)%4); \
	open('$@','w').write('\n'.join('%08x'%w for w in struct.unpack('<%dI'%(len(d)//4),d))+'\n')"

$(SIM)/n_cases.vh: $(ROOT)/rtl/matmul/sim/gen_vectors.py
	mkdir -p $(SIM)
	$(PYTHON) $< --out $(SIM)

# SIM_PREP: optional command run before compiling (e.g. generating test data)
sim: $(SIM)/fw.hex $(SIM)/n_cases.vh
	$(SIM_PREP)
	cd $(SIM) && bash -c 'source $(VIVADO_SETTINGS) >/dev/null && \
	  xvlog -i . -i $(SA) $(addprefix -d ,$(SIM_DEFINES)) -d SIM_D=$(strip $(SIM_D)) $(RTL) $(FW_ROOT)/sim/tb_system.v >xvlog.out 2>&1 || { grep ERROR xvlog.out; exit 1; }; \
	  xelab -debug off tb_system -s tb >xelab.out 2>&1 || { cat xelab.out; exit 1; }; \
	  xsim tb -R' | grep -E "^(TB|ERROR|FATAL)"

clean:
	rm -rf build $(FW).elf $(FW).bin

.PHONY: all sim clean
