# PicoRV32 DDR access test.
#   make        -> ddr_test.bin (copy to the board next to ddr_test.py)
#   make sim    -> run the firmware on picorv32_axi in Vivado xsim
# Uses clang/lld (rv32imc, matches COMPRESSED_ISA/ENABLE_MUL/ENABLE_DIV in
# scripts/pico_processor.tcl), so no riscv gcc toolchain is required.

CC      := clang
OBJCOPY := llvm-objcopy
CFLAGS  := --target=riscv32-unknown-elf -march=rv32imc -mabi=ilp32 \
           -O2 -ffreestanding -fno-builtin -nostdlib -Wall -Wextra
LDFLAGS := -fuse-ld=lld -T link.ld -Wl,--gc-sections

VIVADO_SETTINGS ?= /home/jon/Projects/Vivado/Vivado/2024.1/settings64.sh
RTL := ../../picorv32/picorv32.v

all: ddr_test.bin

ddr_test.elf: start.S ddr_test.c mailbox.h link.ld
	$(CC) $(CFLAGS) $(LDFLAGS) start.S ddr_test.c -o $@

ddr_test.bin: ddr_test.elf
	$(OBJCOPY) -O binary $< $@

sim/ddr_test.hex: ddr_test.bin
	python3 -c "import sys,struct; d=open('$<','rb').read(); d+=b'\0'*(-len(d)%4); \
	open('$@','w').write('\n'.join('%08x'%w for w in struct.unpack('<%dI'%(len(d)//4),d))+'\n')"

sim: sim/ddr_test.hex
	cd sim && bash -c 'source $(VIVADO_SETTINGS) >/dev/null && \
	  xvlog -i .. ../$(RTL) tb_ddr_test.v >/dev/null && \
	  xelab -debug off tb_ddr_test -s tb >/dev/null && \
	  xsim tb -R' | grep -E "^(TB|FATAL|ERROR)"

clean:
	rm -rf ddr_test.elf ddr_test.bin sim/ddr_test.hex sim/xsim.dir sim/*.log sim/*.jou sim/*.pb sim/.Xil

.PHONY: all sim clean
