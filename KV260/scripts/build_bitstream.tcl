# ----------------------------------------------------------------------
# One-shot, non-interactive build of the PicoRV32 + accelerator overlay for
# the Kria KV260 (docs/kv260_upgrade_plan.md K1a / K1b / K1c).
#
#   ./KV260/scripts/build_bitstream.sh [-jobs N] [-sa_d 8|16] [-sa_mhz MHZ] [-proj_dir DIR] [-bd_only]
#
#   -sa_d    array size D (sa_unit parameter); default 8: K1a, the Z1 L2 accelerator
#   -sa_mhz  pl_clk0, the one PL clock; default 50 (K1a; K1b: 100)
#   -bd_only build and validate the block design and write the address map,
#            no synthesis (a quick check of kv260_bd.tcl)
#
# Flow: fresh project (part xck26-sfvc784-2LV-c, board kv260_som) -> IP
# repository RISCV-on-PYNQ-Z1/ip (picorv32_axi) -> kv260_bd.tcl -> wrapper ->
# synth -> impl -> write_bitstream -> KV260/build/output/picorv32.{bit,hwh}
# (+ timing / utilization reports, the address map). PYNQ needs the .bit and
# .hwh to share a basename. No PL I/O, so no pin constraints.
# ----------------------------------------------------------------------

set script_folder [file dirname [file normalize [info script]]]
set kv_root       [file dirname $script_folder]
set repo_root     [file dirname $kv_root]

set part          xck26-sfvc784-2LV-c
set board_part    xilinx.com:kv260_som:part0:1.4
set carrier       xilinx.com:kv260_carrier:som240_1_connector:1.3
set vivado_ver    2024.1
set proj_name     picorv32_kv260
set top_bd        design_1
set out_name      picorv32

set jobs     4
set sa_d     8
set sa_mhz   50
set bd_only  0
set_param general.maxThreads 4
set proj_dir [file join $kv_root build $proj_name]

proc die {msg} { puts "\nERROR: \[kv260_build\] $msg\n"; exit 1 }
proc info_msg {msg} { puts "\nINFO: \[kv260_build\] $msg" }

for {set i 0} {$i < [llength $argv]} {incr i} {
    set arg [lindex $argv $i]
    switch -- $arg {
        -jobs     { set jobs [lindex $argv [incr i]] }
        -sa_d     { set sa_d [lindex $argv [incr i]] }
        -sa_mhz   { set sa_mhz [lindex $argv [incr i]] }
        -proj_dir { set proj_dir [file normalize [lindex $argv [incr i]]] }
        -bd_only  { set bd_only 1 }
        default   { die "Unknown argument '$arg'. Valid: -jobs N, -sa_d 8|16, -sa_mhz MHZ, -proj_dir DIR, -bd_only" }
    }
}
if {![string is integer -strict $jobs] || $jobs < 1} { die "-jobs expects a positive integer, got '$jobs'" }
if {$sa_d != 8 && $sa_d != 16} { die "-sa_d expects 8 or 16, got '$sa_d'" }
if {![string is integer -strict $sa_mhz] || $sa_mhz < 10 || $sa_mhz > 300} { die "-sa_mhz expects 10..300, got '$sa_mhz'" }
if {[string first $vivado_ver [version -short]] == -1} {
    die "the scripts are written for Vivado $vivado_ver, this is Vivado [version -short]"
}
info_msg "KV260: D = $sa_d, pl_clk0 = $sa_mhz MHz"

set out_dir [file join $kv_root build output]

# ------------------------------------------------------------- project
info_msg "Creating project in $proj_dir"
create_project -force $proj_name $proj_dir -part $part
if {[llength [get_board_parts -quiet $board_part]] == 0} {
    die "board part $board_part not installed (Vivado's XHub board store: Kria KV260)"
}
set_property BOARD_PART $board_part [current_project]
if {[llength [get_board_parts -quiet xilinx.com:kv260_carrier:*]] > 0} {
    set_property board_connections [list som240_1_connector $carrier] [current_project]
}
set_property ip_repo_paths [list [file join $repo_root RISCV-on-PYNQ-Z1 ip]] [current_project]
update_ip_catalog

# ---------------------------------------------------------- block design
info_msg "Building block design $top_bd"
source [file join $script_folder kv260_bd.tcl]
set bd_file [get_files -quiet $top_bd.bd]
if {$bd_file eq "" || [llength [get_bd_cells -quiet pico_processor_0/picorv32]] == 0} {
    die "kv260_bd.tcl did not build $top_bd (see messages above)"
}
file mkdir $out_dir
set fh [open [file join $out_dir address_map.txt] w]
foreach space [get_bd_addr_spaces -quiet] {
    foreach seg [get_bd_addr_segs -quiet -of_objects $space] {
        set off [get_property OFFSET $seg]
        if {$off eq ""} continue
        puts $fh [format "%-45s %-55s %s %s" $space [get_property NAME $seg] $off [get_property RANGE $seg]]
    }
}
close $fh
info_msg "address map: [file join $out_dir address_map.txt]"
if {$bd_only} {
    info_msg "Done (-bd_only): block design validated."
    exit 0
}

generate_target all $bd_file
set wrapper [make_wrapper -files $bd_file -top -force]
add_files -norecurse $wrapper
set_property top ${top_bd}_wrapper [current_fileset]
update_compile_order -fileset sources_1

# ------------------------------------------------------------ synthesis
info_msg "Running synthesis ($jobs jobs)"
launch_runs synth_1 -jobs $jobs
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {
    die "Synthesis failed. Log: [get_property DIRECTORY [get_runs synth_1]]/runme.log"
}

# ------------------------------------------------------- implementation
info_msg "Running implementation + write_bitstream ($jobs jobs)"
launch_runs impl_1 -to_step write_bitstream -jobs $jobs
wait_on_run impl_1
set impl_dir [get_property DIRECTORY [get_runs impl_1]]
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {
    die "Implementation failed. Log: $impl_dir/runme.log"
}

# -------------------------------------------------------------- outputs
set bit_file [file join $impl_dir ${top_bd}_wrapper.bit]
set hwh_file [lindex [glob -nocomplain [file join $proj_dir *.gen sources_1 bd $top_bd hw_handoff $top_bd.hwh]] 0]
if {![file exists $bit_file]} { die "Bitstream not found: $bit_file" }
if {$hwh_file eq ""}          { die "Hardware handoff ($top_bd.hwh) not found under $proj_dir" }
file copy -force $bit_file [file join $out_dir $out_name.bit]
file copy -force $hwh_file [file join $out_dir $out_name.hwh]

open_run impl_1
report_timing_summary -file [file join $out_dir timing_summary.rpt]
report_utilization    -file [file join $out_dir utilization.rpt]
report_clocks         -file [file join $out_dir clocks.rpt]
check_timing          -file [file join $out_dir check_timing.rpt]
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
set whs [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -hold]]
close_design

info_msg "Done.\n    [file join $out_dir $out_name.bit]\n    [file join $out_dir $out_name.hwh]\n    setup WNS = $wns ns, hold WHS = $whs ns"
if {($wns ne "" && $wns < 0) || ($whs ne "" && $whs < 0)} {
    puts "WARNING: \[kv260_build\] Timing not met (WNS = $wns, WHS = $whs); see timing_summary.rpt."
}
