################################################################
# PicoRV32 processor subsystem, built as a hierarchical cell.
#
# Sourced by pico_bit.tcl, which calls
#     create_hier_cell_pico_processor <parent> pico_processor_0
# and assigns the addresses. (This used to be a separate block design
# instantiated as a Block Design Container; PYNQ's .hwh parser cannot
# resolve address ranges inside a BDC, so Overlay() failed with
# KeyError: 'S_AXI'.)
#
# Pins:
#   S_AXI_MEM     ARM access to the program BRAM (psBramController)
#   M_AXI_DDR     RISC-V master to PS DDR (-> S_AXI_HP0)
#   riscv_clk     PicoRV32 clock domain (also clocks M_AXI_DDR)
#   riscv_resetn  active-HIGH hold-in-reset from PS GPIO EMIO[0]
#   por_resetn    active-low power-on reset (FCLK_RESET0_N)
#   s_axi_aclk / s_axi_aresetn  clock/reset of S_AXI_MEM
#   irq           PicoRV32 trap
################################################################

set pico_processor_ips {
    cliffordwolf:ip:picorv32_axi:1.0
    xilinx.com:ip:axi_bram_ctrl:4.1
    xilinx.com:ip:axi_interconnect:2.1
    xilinx.com:ip:blk_mem_gen:8.4
    xilinx.com:ip:proc_sys_reset:5.0
}

proc create_hier_cell_pico_processor { parentCell nameHier } {

  if { $parentCell eq "" || $nameHier eq "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2092 -severity "ERROR" "create_hier_cell_pico_processor() - Empty argument(s)!"}
     return
  }

  # Get object for parentCell
  set parentObj [get_bd_cells $parentCell]
  if { $parentObj == "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2090 -severity "ERROR" "Unable to find parent cell <$parentCell>!"}
     return
  }

  # Make sure parentObj is hier blk
  set parentType [get_property TYPE $parentObj]
  if { $parentType ne "hier" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2091 -severity "ERROR" "Parent <$parentObj> has TYPE = <$parentType>. Expected to be <hier>."}
     return
  }

  # Save current instance; Restore later
  set oldCurInst [current_bd_instance .]

  # Set parent object as current
  current_bd_instance $parentObj

  # Create cell and set as current instance
  set hier_obj [create_bd_cell -type hier $nameHier]
  current_bd_instance $hier_obj

  # Create interface pins
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 S_AXI_MEM
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI_DDR

  # Create pins
  create_bd_pin -dir O irq
  create_bd_pin -dir I -type rst por_resetn
  create_bd_pin -dir I -type clk riscv_clk
  create_bd_pin -dir I -type rst riscv_resetn
  create_bd_pin -dir I -type clk s_axi_aclk
  create_bd_pin -dir I -type rst s_axi_aresetn

  # Create instance: picorv32, and set properties
  set picorv32 [ create_bd_cell -type ip -vlnv cliffordwolf:ip:picorv32_axi:1.0 picorv32 ]
  set_property -dict [ list \
   CONFIG.COMPRESSED_ISA {true} \
   CONFIG.ENABLE_DIV {true} \
   CONFIG.ENABLE_MUL {true} \
   CONFIG.ENABLE_PCPI {true} \
   CONFIG.PROGADDR_RESET {0xC0000000} \
   CONFIG.STACKADDR {0xC0002000} \
 ] $picorv32

  # Create instance: riscvAxiInterconnect, and set properties
  set riscvAxiInterconnect [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 riscvAxiInterconnect ]
  set_property -dict [ list \
   CONFIG.NUM_MI {2} \
 ] $riscvAxiInterconnect

  # Create instance: psBramController, and set properties
  set psBramController [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_bram_ctrl:4.1 psBramController ]
  set_property -dict [ list \
   CONFIG.ECC_TYPE {0} \
   CONFIG.SINGLE_PORT_BRAM {1} \
   CONFIG.USE_ECC {0} \
 ] $psBramController

  # Create instance: riscvBram, and set properties
  set riscvBram [ create_bd_cell -type ip -vlnv xilinx.com:ip:blk_mem_gen:8.4 riscvBram ]
  set_property -dict [ list \
   CONFIG.EN_SAFETY_CKT {false} \
   CONFIG.Enable_B {Use_ENB_Pin} \
   CONFIG.Memory_Type {True_Dual_Port_RAM} \
   CONFIG.Port_B_Clock {100} \
   CONFIG.Port_B_Enable_Rate {100} \
   CONFIG.Port_B_Write_Rate {50} \
   CONFIG.Use_RSTB_Pin {true} \
 ] $riscvBram

  # Create instance: riscvBramController, and set properties
  set riscvBramController [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_bram_ctrl:4.1 riscvBramController ]
  set_property -dict [ list \
   CONFIG.ECC_TYPE {0} \
   CONFIG.PROTOCOL {AXI4LITE} \
   CONFIG.SINGLE_PORT_BRAM {1} \
 ] $riscvBramController

  # Create instance: riscvReset, and set properties
  set riscvReset [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 riscvReset ]
  set_property -dict [ list \
   CONFIG.C_AUX_RESET_HIGH {1} \
   CONFIG.C_AUX_RST_WIDTH {1} \
   CONFIG.C_EXT_RST_WIDTH {1} \
 ] $riscvReset

  # Create interface connections
  connect_bd_intf_net -intf_net Conn2 [get_bd_intf_pins S_AXI_MEM] [get_bd_intf_pins psBramController/S_AXI]
  connect_bd_intf_net -intf_net picorv32_mem_axi [get_bd_intf_pins picorv32/mem_axi] [get_bd_intf_pins riscvAxiInterconnect/S00_AXI]
  connect_bd_intf_net -intf_net riscvAxiInterconnect_M00_AXI [get_bd_intf_pins riscvAxiInterconnect/M00_AXI] [get_bd_intf_pins riscvBramController/S_AXI]
  connect_bd_intf_net -intf_net riscvAxiInterconnect_M01_AXI [get_bd_intf_pins riscvAxiInterconnect/M01_AXI] [get_bd_intf_pins M_AXI_DDR]
  connect_bd_intf_net -intf_net psBramController_BRAM_PORTA [get_bd_intf_pins psBramController/BRAM_PORTA] [get_bd_intf_pins riscvBram/BRAM_PORTB]
  connect_bd_intf_net -intf_net riscvBramController_BRAM_PORTA [get_bd_intf_pins riscvBram/BRAM_PORTA] [get_bd_intf_pins riscvBramController/BRAM_PORTA]

  # Create port connections
  connect_bd_net -net aux_reset_in_1 [get_bd_pins riscv_resetn] [get_bd_pins riscvReset/aux_reset_in]
  connect_bd_net -net clk_in1_1 [get_bd_pins s_axi_aclk] [get_bd_pins psBramController/s_axi_aclk]
  connect_bd_net -net ext_reset_in_1 [get_bd_pins por_resetn] [get_bd_pins riscvReset/ext_reset_in]
  connect_bd_net -net picorv32_axi_0_trap [get_bd_pins irq] [get_bd_pins picorv32/trap]
  connect_bd_net -net riscvReset_peripheral_aresetn [get_bd_pins picorv32/resetn] [get_bd_pins riscvBramController/s_axi_aresetn] [get_bd_pins riscvReset/peripheral_aresetn] [get_bd_pins riscvAxiInterconnect/S00_ARESETN] [get_bd_pins riscvAxiInterconnect/M00_ARESETN] [get_bd_pins riscvAxiInterconnect/M01_ARESETN]
  connect_bd_net -net riscvReset_interconnect_aresetn [get_bd_pins riscvReset/interconnect_aresetn] [get_bd_pins riscvAxiInterconnect/ARESETN]
  connect_bd_net -net riscv_clk_1subprocessorClk [get_bd_pins riscv_clk] [get_bd_pins picorv32/clk] [get_bd_pins riscvBramController/s_axi_aclk] [get_bd_pins riscvReset/slowest_sync_clk] [get_bd_pins riscvAxiInterconnect/ACLK] [get_bd_pins riscvAxiInterconnect/S00_ACLK] [get_bd_pins riscvAxiInterconnect/M00_ACLK] [get_bd_pins riscvAxiInterconnect/M01_ACLK]
  connect_bd_net -net s_axi_aresetn_1 [get_bd_pins s_axi_aresetn] [get_bd_pins psBramController/s_axi_aresetn]

  # Restore current instance
  current_bd_instance $oldCurInst
}
