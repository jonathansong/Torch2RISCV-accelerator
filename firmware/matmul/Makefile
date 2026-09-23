# Phase 3 PicoRV32 firmware: batch 8x8x8 matmuls on matmul_unit via CSRs.
#   make        -> matmul_fw.bin (load into the program BRAM; see driver/pynq_matmul.py)
#   make sim    -> system test: firmware on picorv32_axi + matmul_unit in xsim
# clang/lld, rv32imc (matches the overlay's PicoRV32 configuration).

CC      := clang
OBJCOPY := llvm-objcopy
CFLAGS  := --target=riscv32-unknown-elf -march=rv32imc -mabi=ilp32 \
           -O2 -ffreestanding -fno-builtin -nostdlib -Wall -Wextra
LDFLAGS := -fuse-ld=lld -T link.ld -Wl,--gc-sections

VIVADO_SETTINGS ?= /home/jon/Projects/Vivado/Vivado/2024.1/settings64.sh
PYTHON          ?= python3

ROOT := $(abspath ../..)
RTL  := $(ROOT)/RISCV-on-PYNQ-Z1/picorv32/picorv32.v \
        $(ROOT)/rtl/matmul/systolic_array.v $(ROOT)/rtl/matmul/matmul_unit.v
SIM  := build/sim

all: matmul_fw.bin

matmul_fw.elf: start.S matmul_fw.c mailbox.h matmul_csr.h link.ld
	$(CC) $(CFLAGS) $(LDFLAGS) start.S matmul_fw.c -o $@

matmul_fw.bin: matmul_fw.elf
	$(OBJCOPY) -O binary $< $@

$(SIM)/matmul_fw.hex: matmul_fw.bin
	mkdir -p $(SIM)
	$(PYTHON) -c "import struct; d=open('$<','rb').read(); d+=b'\0'*(-len(d)%4); \
	open('$@','w').write('\n'.join('%08x'%w for w in struct.unpack('<%dI'%(len(d)//4),d))+'\n')"

$(SIM)/n_cases.vh: $(ROOT)/rtl/matmul/sim/gen_vectors.py
	mkdir -p $(SIM)
	$(PYTHON) $< --out $(SIM)

sim: $(SIM)/matmul_fw.hex $(SIM)/n_cases.vh
	cd $(SIM) && bash -c 'source $(VIVADO_SETTINGS) >/dev/null && \
	  xvlog -i . $(RTL) $(abspath sim/tb_system.v) >xvlog.out 2>&1 || { cat xvlog.out; exit 1; }; \
	  xelab -debug off tb_system -s tb >xelab.out 2>&1 || { cat xelab.out; exit 1; }; \
	  xsim tb -R' | grep -E "^(TB|ERROR|FATAL)"

clean:
	rm -rf build matmul_fw.elf matmul_fw.bin

.PHONY: all sim clean
