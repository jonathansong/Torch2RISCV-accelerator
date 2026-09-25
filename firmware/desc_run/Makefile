# Descriptor-list runner: submits a list the ARM built in DDR (mat_submit).
FW   := desc_run_fw
SRCS := desc_run_fw.c
SIM_DEFINES := DESC_TEST
include ../common.mk
# co-simulation cases built with the driver's DescList (needs NumPy in PYTHON)
SIM_PREP = $(PYTHON) $(CURDIR)/gen_desc_cases.py --d $(SIM_D) --out $(SIM) >/dev/null
