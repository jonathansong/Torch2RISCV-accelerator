# RISC-V backend test (docs/riscv_backend_design.md §5): one dispatch prepared
# by compiler/tests/rv_dispatch.py (RVB = its output directory), run on the
# system testbench (RVB_TEST) as its descriptor list and over PCPI.
#   make sim RVB=../../build/rvb/smol_7 [RVB_CODE=ddr] [SIM_CPU_DDR_LAT=40]
# Needs D = 16 with the GEMV unit (SIM_D=16 SIM_DMA_W=128 SIM_GEMV=1, the defaults here).
FW_ROOT := $(abspath ..)
ROOT    := $(abspath ../..)
RVB     ?= $(ROOT)/build/rvb/smol_7
RVB_CODE ?= bram
SIM_D ?= 16
SIM_DMA_W ?= 128
SIM_GEMV ?= 1
SIM_CPU_DDR_LAT ?= 0
CC      := clang
OBJCOPY := llvm-objcopy
CFLAGS  := --target=riscv32-unknown-elf -march=rv32imc -mabi=ilp32 -O2 -ffreestanding -fno-builtin -nostdlib \
           -Wall -I$(FW_ROOT)/include -I$(RVB) $(if $(filter ddr,$(RVB_CODE)),-DRVB_CODE_DDR)
VIVADO_SETTINGS ?= /home/jon/Projects/Vivado/Vivado/2024.1/settings64.sh
PYTHON ?= python3
SA  := $(ROOT)/rtl/sysarray
RTL := $(ROOT)/rtl/picorv32/picorv32.v $(wildcard $(SA)/*.v)
SIM := $(RVB)/sim_$(RVB_CODE)
DDR_BYTES = $(shell sed -n 's/#define RVB_DDR_BYTES //p' $(RVB)/rvb_args.h)
CODE_ADDR = $(shell printf '0x%x' $$((0x18000000 + $(DDR_BYTES))))

ifeq ($(RVB_CODE),ddr)
LDFLAGS := -fuse-ld=lld -T link_ddr.ld -Wl,--defsym=RVB_CODE_ADDR=$(CODE_ADDR) -Wl,--gc-sections
else
LDFLAGS := -fuse-ld=lld -T $(FW_ROOT)/common/link.ld -Wl,--gc-sections
endif

# rv_dispatch.py --code compiled: rvb_gen.c calls the export's generated function (rvb_gen.ll,
# docs/riscv_backend_design.md §9), linked with its runtime. In DDR both go to .rvb_gen
# (link_ddr.ld, by object name): position independent (medany: PC-relative), no jump tables,
# no linker relaxation (gp-relative accesses would tie the code to its link address).
RVLL := $(wildcard $(RVB)/rvb_gen.ll)
RVFLAGS := --target=riscv32-unknown-elf -march=rv32imc -mabi=ilp32 -O2 -ffreestanding -fno-builtin -nostdlib \
           -mcmodel=medany -fno-jump-tables -mno-relax -fno-asynchronous-unwind-tables -fno-unwind-tables -Wno-override-module -I$(FW_ROOT)/include
define rvobjs
$(if $(RVLL),$(CC) $(RVFLAGS) -c $(RVLL) -o $(1)/rvb_gen_ll.o && \
  $(CC) $(RVFLAGS) -c $(FW_ROOT)/rv/sa_rv_rt.c -o $(1)/sa_rv_rt.o,true)
endef
RVOBJS = $(if $(RVLL),$(1)/rvb_gen_ll.o $(1)/sa_rv_rt.o)

sim:
	mkdir -p $(SIM)
	$(call rvobjs,$(SIM))
	$(CC) $(CFLAGS) $(LDFLAGS) -Wl,--no-relax $(FW_ROOT)/common/start.S rvb_fw.c $(RVB)/rvb_gen.c $(call RVOBJS,$(SIM)) -o $(SIM)/rvb_fw.elf
	$(OBJCOPY) -O binary $(if $(filter ddr,$(RVB_CODE)),--remove-section=.rvb_gen) $(SIM)/rvb_fw.elf $(SIM)/fw.bin
	$(PYTHON) -c "import struct; d=open('$(SIM)/fw.bin','rb').read(); d+=b'\0'*(-len(d)%4); \
	open('$(SIM)/fw.hex','w').write('\n'.join('%08x'%w for w in struct.unpack('<%dI'%(len(d)//4),d))+'\n'); \
	print('BRAM image', len(d), 'bytes')"
ifeq ($(RVB_CODE),ddr)
	$(OBJCOPY) -O binary --only-section=.rvb_gen $(SIM)/rvb_fw.elf $(SIM)/code.bin
	$(PYTHON) -c "import struct; d=open('$(SIM)/code.bin','rb').read(); d+=b'\0'*(-len(d)%4); \
	open('$(SIM)/rvb_code.hex','w').write('@%x\n'%($(DDR_BYTES)//4)+'\n'.join('%08x'%w for w in struct.unpack('<%dI'%(len(d)//4),d))+'\n'); \
	print('DDR code', len(d), 'bytes')"
else
	llvm-size $(SIM)/rvb_fw.elf | tail -1
endif
	cp $(RVB)/rvb_ddr.hex $(RVB)/rvb_expect.hex $(RVB)/rvb_cmp.txt $(SIM)/
	$(PYTHON) $(SA)/sim/gen_vectors.py --out $(SIM) >/dev/null
	cd $(SIM) && bash -c 'source $(VIVADO_SETTINGS) >/dev/null && \
	  xvlog -i . -i $(SA) -d RVB_TEST $(if $(filter ddr,$(RVB_CODE)),-d RVB_DDR_CODE) -d SIM_D=$(SIM_D) -d SIM_DMA_W=$(SIM_DMA_W) -d SIM_GEMV=$(SIM_GEMV) \
	    -d SIM_DDR_BYTES=$$(( $(DDR_BYTES) + 65536 )) -d SIM_CPU_DDR_LAT=$(SIM_CPU_DDR_LAT) -d SIM_RVB_CODE=$$(( $(CODE_ADDR) )) $(RTL) $(FW_ROOT)/sim/tb_system.v >xvlog.out 2>&1 || { grep ERROR xvlog.out; exit 1; }; \
	  xelab -debug off tb_system -s tb >xelab.out 2>&1 || { cat xelab.out; exit 1; }; \
	  xsim tb -R' | grep -E "^(TB|ERROR|FATAL)"

# the board's images (board_rvb.py): fw_bram.bin (rvb_gen in the BRAM image), fw_ddr.bin +
# code.bin (rvb_gen loaded at the window's end), into OUT
OUT ?= $(RVB)/board
bundle:
	mkdir -p $(OUT)
	$(call rvobjs,$(OUT))
	$(CC) $(CFLAGS) -fuse-ld=lld -T $(FW_ROOT)/common/link.ld -Wl,--gc-sections -Wl,--no-relax $(FW_ROOT)/common/start.S rvb_fw.c $(RVB)/rvb_gen.c $(call RVOBJS,$(OUT)) -o $(OUT)/fw_bram.elf
	$(OBJCOPY) -O binary $(OUT)/fw_bram.elf $(OUT)/fw_bram.bin
	$(CC) $(CFLAGS) -DRVB_CODE_DDR -fuse-ld=lld -T link_ddr.ld -Wl,--defsym=RVB_CODE_ADDR=$(CODE_ADDR) -Wl,--gc-sections \
	  -Wl,--no-relax $(FW_ROOT)/common/start.S rvb_fw.c $(RVB)/rvb_gen.c $(call RVOBJS,$(OUT)) -o $(OUT)/fw_ddr.elf
	$(OBJCOPY) -O binary --remove-section=.rvb_gen $(OUT)/fw_ddr.elf $(OUT)/fw_ddr.bin
	$(OBJCOPY) -O binary --only-section=.rvb_gen $(OUT)/fw_ddr.elf $(OUT)/code.bin
	@ls -l $(OUT)/fw_bram.bin $(OUT)/fw_ddr.bin $(OUT)/code.bin | awk '{print $$5, $$9}'

.PHONY: sim bundle
