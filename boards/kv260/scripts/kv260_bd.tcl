# ----------------------------------------------------------------------
# Block design design_1 for the Kria KV260 (docs/kv260_upgrade_plan.md K1a):
# the PL of the PYNQ-Z1 L2 overlay (pico_bit.tcl on the pynq-z1 branch)
# on the K26 SOM's Zynq UltraScale+ PS, with the accelerator unchanged.
#
# Sourced by build_bitstream.tcl (project already created, IP repository
# registered; variables ::sa_d, ::sa_mhz, ::dma_w, ::vefp_nb, ::gemv, ::nports, ::m1_hp,
# ::pico_port, ::dma_ports: the PS slave port of PicoRV32 and of m0, m1, ...).
#
#   zynq_ultra_ps_e_0   board preset (DDR4, MIO), then:
#     M_AXI_HPM0_FPD    32-bit, ARM -> psAxiInterconnect -> program BRAM, interrupt controller
#     S_AXI_HP0_FPD     64-bit, PicoRV32's DDR port (the rings in DDR)   [Z1: HP0]
#     S_AXI_HP1_FPD     ::dma_w bits (64; K2a 128), the accelerator's DMA (m0_axi)  [Z1: HP2 + AXI4->AXI3]
#     S_AXI_HP<m1_hp>_FPD  ::nports = 2 (K2b G3): the second DMA port (m1_axi), HP3 by default:
#                       HP1 and HP2 share one DDR controller port (S4), HP3 has its own (S5;
#                       UG1085 "PS Interconnect"); -m1_hp 2 for the comparison
#     ::nports = 4 (G5): m2_axi on HP2, m3_axi on HPC0 (through the CCI to the DDR
#                       controller's S1 / S2; used non-coherently) or, -m3_port hp0, on HP0
#                       with PicoRV32 moved to HPC0
#     pl_clk0           ::sa_mhz MHz, the only PL clock (PicoRV32, accelerator, AXI)
#     pl_ps_irq0[0]     the interrupt controller (PicoRV32 trap, matmul_0/notify_irq)
#     emio_gpio_o[0]    RISC-V reset (1 = hold), as EMIO[0] on the Z1 (2 EMIO pins: xlslice needs >= 2)
#   pico_processor_0    the Z1 hierarchy as is (pico_processor.tcl)
#   matmul_0            sa_unit (rtl/sysarray), D = ::sa_d (all PE columns on DSPs), 128 KB per SPAD, 256 KB ACC
#
# Differences from the Z1 design: one clock domain (the Z1's clk_wiz with
# its AXI reconfiguration port is not used by the driver); no PL I/O (the
# Z1's LED / button / Arduino / PMOD ports were unused); no AXI3 converter
# (the ZynqMP HP ports are AXI4). The address map keeps the Z1's offsets
# inside the HPM0_FPD window (0xA000_0000) on the ARM side and is unchanged
# on the RISC-V / accelerator side, except that DDR spans the low 2 GB.
# ----------------------------------------------------------------------

source [file join [file dirname [file normalize [info script]]] pico_processor.tcl]

# IP versions of Vivado 2024.1 (the Z1 scripts' versions; get_ipdefs also
# lists unsupported older ones, e.g. axi_interconnect 1.7)
proc ipdef {name} {
    set v [dict get {zynq_ultra_ps_e 3.5 axi_interconnect 2.1 axi_intc 4.1 xlconcat 2.1
                     proc_sys_reset 5.0 xlslice 1.0} $name]
    return xilinx.com:ip:$name:$v
}

set design_name design_1
create_bd_design $design_name
current_bd_design $design_name

# ---------------------------------------------------------------- PS
# PS slave ports: S_AXI_GP index, interface, clock pin
set ::ps_ports {HPC0 {0 S_AXI_HPC0_FPD saxihpc0_fpd_aclk} HP0 {2 S_AXI_HP0_FPD saxihp0_fpd_aclk}
              HP1 {3 S_AXI_HP1_FPD saxihp1_fpd_aclk} HP2 {4 S_AXI_HP2_FPD saxihp2_fpd_aclk}
              HP3 {5 S_AXI_HP3_FPD saxihp3_fpd_aclk}}
proc ps_port_used {p} { expr {$p eq $::pico_port || $p in $::dma_ports} }
proc ps_port {p field} { lindex [dict get $::ps_ports $p] [lsearch {gp intf clk} $field] }
if {[llength $::dma_ports] != $::nports} { error "dma_ports '$::dma_ports' for NPORTS $::nports" }
set ps [create_bd_cell -type ip -vlnv [ipdef zynq_ultra_ps_e] zynq_ultra_ps_e_0]
if {[get_property BOARD_PART [current_project]] ne ""} {
    apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} $ps
}
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__MAXIGP0__DATA_WIDTH {32} \
    CONFIG.PSU__USE__M_AXI_GP1 {0} \
    CONFIG.PSU__USE__M_AXI_GP2 {0} \
    CONFIG.PSU__USE__S_AXI_GP0 [ps_port_used HPC0] \
    CONFIG.PSU__USE__S_AXI_GP2 [ps_port_used HP0] \
    CONFIG.PSU__USE__S_AXI_GP3 [ps_port_used HP1] \
    CONFIG.PSU__USE__S_AXI_GP4 [ps_port_used HP2] \
    CONFIG.PSU__USE__S_AXI_GP5 [ps_port_used HP3] \
    CONFIG.PSU__FPGA_PL0_ENABLE {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__SRCSEL {IOPLL} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $::sa_mhz \
    CONFIG.PSU__FPGA_PL1_ENABLE {0} \
    CONFIG.PSU__USE__IRQ0 {1} \
    CONFIG.PSU__GPIO_EMIO__PERIPHERAL__ENABLE {1} \
    CONFIG.PSU__GPIO_EMIO__PERIPHERAL__IO {2} \
] $ps

# ---------------------------------------------------------------- PL cells
set psAxiInterconnect [create_bd_cell -type ip -vlnv [ipdef axi_interconnect] psAxiInterconnect]
set_property CONFIG.NUM_MI {2} $psAxiInterconnect

set psInterruptController [create_bd_cell -type ip -vlnv [ipdef axi_intc] psInterruptController]

set irqConcat [create_bd_cell -type ip -vlnv [ipdef xlconcat] irqConcat]
set_property CONFIG.NUM_PORTS {2} $irqConcat      ;# In0: PicoRV32 trap; In1: matmul_0/notify_irq

set porReset [create_bd_cell -type ip -vlnv [ipdef proc_sys_reset] porReset]
set_property -dict [list CONFIG.C_AUX_RST_WIDTH {1} CONFIG.C_EXT_RST_WIDTH {1}] $porReset

set gpio_w [expr {[get_property LEFT [get_bd_pins $ps/emio_gpio_o]] + 1}]
set resetSlice [create_bd_cell -type ip -vlnv [ipdef xlslice] resetSlice]
set_property -dict [list CONFIG.DIN_WIDTH $gpio_w CONFIG.DIN_FROM {0} CONFIG.DIN_TO {0}] $resetSlice

create_hier_cell_pico_processor [current_bd_instance .] pico_processor_0

set rtl_dir [file normalize [file join [file dirname [info script]] .. .. .. rtl sysarray]]
if {[get_files -quiet sa_unit.v] eq ""} {
    add_files -norecurse [concat [glob [file join $rtl_dir *.v]] [glob [file join $rtl_dir *.vh]]]
    set_property file_type {Verilog Header} [get_files -filter {NAME =~ *.vh}]
    set_property include_dirs $rtl_dir [get_filesets sources_1]
    update_compile_order -fileset sources_1
}
set matmul_0 [create_bd_cell -type module -reference sa_unit matmul_0]
# (the module reference freezes the defaults at D = 8: the depths that derive
# from D are set explicitly, as on the Z1). Every PE column on DSPs
# (DSP_COLS = D): the K26 has 1248 DSP48E2; the Z1's D = 16 builds put only 8
# columns on DSPs to fit its 220.
set_property -dict [list CONFIG.D $::sa_d \
                         CONFIG.DSP_COLS $::sa_d \
                         CONFIG.DMA_W $::dma_w \
                         CONFIG.VEFP_NB $::vefp_nb \
                         CONFIG.GEMV $::gemv \
                         CONFIG.NPORTS $::nports \
                         CONFIG.SPAD_WORDS [expr {131072 / $::sa_d}] \
                         CONFIG.ACC_WORDS [expr {262144 / (4 * $::sa_d)}]] $matmul_0

# ---------------------------------------------------------------- connections
connect_bd_intf_net [get_bd_intf_pins $ps/M_AXI_HPM0_FPD] [get_bd_intf_pins psAxiInterconnect/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins psAxiInterconnect/M00_AXI] [get_bd_intf_pins psInterruptController/s_axi]
connect_bd_intf_net [get_bd_intf_pins psAxiInterconnect/M01_AXI] [get_bd_intf_pins pico_processor_0/S_AXI_MEM]
# each used PS slave port: its data width once the port exists (PicoRV32 64 bits, DMA ::dma_w)
set_property CONFIG.PSU__SAXIGP[ps_port $::pico_port gp]__DATA_WIDTH 64 $ps
foreach p $::dma_ports { set_property CONFIG.PSU__SAXIGP[ps_port $p gp]__DATA_WIDTH $::dma_w $ps }
connect_bd_intf_net [get_bd_intf_pins pico_processor_0/M_AXI_DDR] [get_bd_intf_pins $ps/[ps_port $::pico_port intf]]
connect_bd_intf_net [get_bd_intf_pins pico_processor_0/M_AXI_PERIPH] [get_bd_intf_pins matmul_0/s_axi]
connect_bd_intf_net [get_bd_intf_pins pico_processor_0/PCPI] [get_bd_intf_pins matmul_0/pcpi]
foreach p $::dma_ports i [lsearch -all $::dma_ports *] {
    connect_bd_intf_net [get_bd_intf_pins matmul_0/m${i}_axi] [get_bd_intf_pins $ps/[ps_port $p intf]]
}

# the one PL clock
connect_bd_net [get_bd_pins $ps/pl_clk0] \
    [get_bd_pins $ps/maxihpm0_fpd_aclk] \
    {*}[lmap p [concat [list $::pico_port] $::dma_ports] {get_bd_pins $ps/[ps_port $p clk]}] \
    [get_bd_pins psAxiInterconnect/ACLK] [get_bd_pins psAxiInterconnect/S00_ACLK] \
    [get_bd_pins psAxiInterconnect/M00_ACLK] [get_bd_pins psAxiInterconnect/M01_ACLK] \
    [get_bd_pins psInterruptController/s_axi_aclk] [get_bd_pins porReset/slowest_sync_clk] \
    [get_bd_pins pico_processor_0/s_axi_aclk] [get_bd_pins pico_processor_0/riscv_clk] \
    [get_bd_pins matmul_0/aclk]

# resets: power-on (PS pl_resetn0) for the ARM side; the RISC-V side's reset
# also follows EMIO[0] (pico_processor's riscvReset, aux reset active high)
connect_bd_net [get_bd_pins $ps/pl_resetn0] [get_bd_pins porReset/ext_reset_in] [get_bd_pins pico_processor_0/por_resetn]
connect_bd_net [get_bd_pins porReset/interconnect_aresetn] [get_bd_pins psAxiInterconnect/ARESETN]
connect_bd_net [get_bd_pins porReset/peripheral_aresetn] \
    [get_bd_pins psAxiInterconnect/S00_ARESETN] [get_bd_pins psAxiInterconnect/M00_ARESETN] \
    [get_bd_pins psAxiInterconnect/M01_ARESETN] [get_bd_pins psInterruptController/s_axi_aresetn] \
    [get_bd_pins pico_processor_0/s_axi_aresetn]
connect_bd_net [get_bd_pins $ps/emio_gpio_o] [get_bd_pins resetSlice/Din]
connect_bd_net [get_bd_pins resetSlice/Dout] [get_bd_pins pico_processor_0/riscv_resetn]
connect_bd_net [get_bd_pins pico_processor_0/periph_aresetn] [get_bd_pins matmul_0/aresetn]

# interrupts
connect_bd_net [get_bd_pins pico_processor_0/irq] [get_bd_pins irqConcat/In0]
connect_bd_net [get_bd_pins matmul_0/notify_irq] [get_bd_pins irqConcat/In1]
connect_bd_net [get_bd_pins irqConcat/dout] [get_bd_pins psInterruptController/intr]
connect_bd_net [get_bd_pins psInterruptController/irq] [get_bd_pins $ps/pl_ps_irq0]

# ---------------------------------------------------------------- addresses
# a slave port's DDR segment (HPx_DDR_LOW: the low 2 GB)
proc ddr_low_seg {pin} {
    set s [get_bd_addr_segs -of_objects [get_bd_intf_pins $pin] -filter {NAME =~ *DDR_LOW}]
    if {$s eq ""} { error "no DDR_LOW segment on $pin" }
    return $s
}
set ps_data [get_bd_addr_spaces $ps/Data]
# ARM side: the Z1's offsets inside the HPM0_FPD window
assign_bd_address -offset 0xA0010000 -range 0x00002000 -target_address_space $ps_data \
    [get_bd_addr_segs pico_processor_0/psBramController/S_AXI/Mem0] -force
assign_bd_address -offset 0xA0020000 -range 0x00010000 -target_address_space $ps_data \
    [get_bd_addr_segs psInterruptController/S_AXI/Reg] -force
# RISC-V: DDR identity-mapped (low 2 GB, below the CSRs), CSRs at 0x8000_0000,
# program BRAM at 0xC000_0000 (reset vector) - as on the Z1
set rv [get_bd_addr_spaces pico_processor_0/picorv32/mem_axi]
assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space $rv \
    [ddr_low_seg $ps/[ps_port $::pico_port intf]] -force
assign_bd_address -offset 0xC0000000 -range 0x00002000 -target_address_space $rv \
    [get_bd_addr_segs pico_processor_0/riscvBramController/S_AXI/Mem0] -force
assign_bd_address -offset 0x80000000 -range 0x00001000 -target_address_space $rv [get_bd_addr_segs matmul_0/s_axi/reg0] -force
# accelerator DMA: DDR identity-mapped (low 2 GB)
foreach p $::dma_ports i [lsearch -all $::dma_ports *] {
    assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces matmul_0/m${i}_axi] \
        [ddr_low_seg $ps/[ps_port $p intf]] -force
}

validate_bd_design
save_bd_design
