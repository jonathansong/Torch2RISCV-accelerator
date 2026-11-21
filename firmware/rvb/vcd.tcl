# xsim: log the sa_unit fetch / scheduler / engine handshakes of tb_system (RVB_TEST) to wave.vcd
# (docs/e2e_walkthrough.md §9); run in a sim directory after xelab -debug typical tb_system -s tbdbg:
#   xsim tbdbg -tclbatch <this file>
open_vcd wave.vcd
log_vcd [get_objects /tb_system/mm/aclk /tb_system/mm/fetch_busy /tb_system/mm/f_valid /tb_system/mm/f_ready /tb_system/mm/in_valid /tb_system/mm/in_ready /tb_system/mm/ld_v /tb_system/mm/ld_r /tb_system/mm/ld_done /tb_system/mm/st_v /tb_system/mm/st_r /tb_system/mm/st_done /tb_system/mm/ve_v /tb_system/mm/ve_r /tb_system/mm/ve_done /tb_system/mm/ex_v /tb_system/mm/ex_done /tb_system/mm/sched_idle /tb_system/mm/m0_axi_arvalid /tb_system/mm/m0_axi_rvalid /tb_system/mm/m0_axi_rready /tb_system/mm/m0_axi_awvalid /tb_system/mm/m0_axi_wvalid /tb_system/mm/fetch_reg_we]
run all
close_vcd
quit
