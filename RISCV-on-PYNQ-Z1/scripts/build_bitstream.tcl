# ----------------------------------------------------------------------
# One-shot, non-interactive build of the PicoRV32 overlay for PYNQ-Z1.
#
# Usage (from anywhere):
#   vivado -mode batch -source scripts/build_bitstream.tcl \
#          -tclargs [-jobs N] [-proj_dir DIR] [-allow_unplaced_io]
#
# Or simply:  ./scripts/build_bitstream.sh [same options]
#
# Flow:
#   1. create a fresh project (DIR, default <repo>/build/picorv32_z1)
#   2. register <repo>/ip as the IP repository
#   3. source pico_bit.tcl        (BD design_1: PS7 + clk + intc, plus the
#                                  PicoRV32 hierarchy from pico_processor.tcl)
#   4. make the HDL wrapper, add constrs/PYNQ-Z1.xdc
#   5. synth -> check I/O placement -> impl -> write_bitstream
#   6. copy <repo>/build/output/picorv32.{bit,hwh} (+ reports)
#
# PYNQ needs the .bit and .hwh to share a basename, so the outputs are
# named picorv32.bit / picorv32.hwh and can be copied to the board as-is.
# ----------------------------------------------------------------------

set script_folder [file dirname [file normalize [info script]]]
set repo_root     [file dirname $script_folder]

set part          xc7z020clg400-1
set board_part    www.digilentinc.com:pynq-z1:part0:1.0
set vivado_ver    2024.1
set proj_name     picorv32_z1
set top_bd        design_1
set out_name      picorv32

set jobs              4
set_param general.maxThreads 4    ;# main Vivado process; runs use -jobs
set proj_dir          [file join $repo_root build $proj_name]
set allow_unplaced_io 0

proc die {msg} {
    puts "\nERROR: \[build_bitstream\] $msg\n"
    exit 1
}

proc info_msg {msg} {
    puts "\nINFO: \[build_bitstream\] $msg"
}

# ---------------------------------------------------------------- args
for {set i 0} {$i < [llength $argv]} {incr i} {
    set arg [lindex $argv $i]
    switch -- $arg {
        -jobs              { set jobs [lindex $argv [incr i]] }
        -proj_dir          { set proj_dir [file normalize [lindex $argv [incr i]]] }
        -allow_unplaced_io { set allow_unplaced_io 1 }
        default            { die "Unknown argument '$arg'. Valid: -jobs N, -proj_dir DIR, -allow_unplaced_io" }
    }
}
if {![string is integer -strict $jobs] || $jobs < 1} {
    die "-jobs expects a positive integer, got '$jobs'"
}

# The generated BD scripts silently `return` (instead of erroring) on a
# version mismatch, so check up front and fail loudly.
if {[string first $vivado_ver [version -short]] == -1} {
    die "pico_processor.tcl / pico_bit.tcl were generated for Vivado $vivado_ver,\
         but this is Vivado [version -short]."
}

set out_dir [file join $repo_root build output]
set xdc     [file join $repo_root constrs PYNQ-Z1.xdc]
if {![file exists $xdc]} { die "Constraint file not found: $xdc" }

# ------------------------------------------------------------- project
info_msg "Creating project in $proj_dir"
create_project -force $proj_name $proj_dir -part $part

if {[llength [get_board_parts -quiet $board_part]] > 0} {
    set_property BOARD_PART $board_part [current_project]
} else {
    # PS7 settings are fully spelled out in pico_bit.tcl, so the board
    # files are a convenience, not a requirement.
    puts "WARNING: \[build_bitstream\] Board part $board_part not installed;\
          continuing with bare part $part."
}

set_property ip_repo_paths [list [file join $repo_root ip]] [current_project]
update_ip_catalog

# ---------------------------------------------------------- block designs
info_msg "Building block design $top_bd"
source [file join $script_folder pico_bit.tcl]
set bd_file [get_files -quiet $top_bd.bd]
if {$bd_file eq "" || [llength [get_bd_cells -quiet pico_processor_0/picorv32]] == 0} {
    die "pico_bit.tcl did not build $top_bd (see messages above)."
}

# pico_bit.tcl already validates and saves; generate the output products
# and the wrapper here.
generate_target all $bd_file
set wrapper [make_wrapper -files $bd_file -top -force]
add_files -norecurse $wrapper
set_property top ${top_bd}_wrapper [current_fileset]
update_compile_order -fileset sources_1

add_files -fileset constrs_1 -norecurse $xdc

# ------------------------------------------------------------ synthesis
info_msg "Running synthesis ($jobs jobs)"
launch_runs synth_1 -jobs $jobs
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {
    die "Synthesis failed. Log: [get_property DIRECTORY [get_runs synth_1]]/runme.log"
}

# Every top-level port needs a pin, otherwise write_bitstream stops on
# DRC UCIO-1/NSTD-1 after a full implementation. Check now instead.
open_run synth_1 -name synth_1
set unplaced {}
foreach port [get_ports] {
    if {[get_property PACKAGE_PIN $port] eq ""} { lappend unplaced $port }
}
close_design

if {[llength $unplaced] > 0} {
    set msg "[llength $unplaced] top-level port(s) have no PACKAGE_PIN in $xdc:\n    [join $unplaced "\n    "]"
    if {!$allow_unplaced_io} {
        die "$msg\nFix the XDC port names, or rerun with -allow_unplaced_io to\
             downgrade DRC UCIO-1/NSTD-1 (leaves those pins unconstrained)."
    }
    puts "WARNING: \[build_bitstream\] $msg"
    set hook [file join $proj_dir allow_unplaced_io.tcl]
    set fh [open $hook w]
    puts $fh "set_property SEVERITY {Warning} \[get_drc_checks {NSTD-1 UCIO-1}\]"
    close $fh
    set_property STEPS.WRITE_BITSTREAM.TCL.PRE $hook [get_runs impl_1]
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

file mkdir $out_dir
file copy -force $bit_file [file join $out_dir $out_name.bit]
file copy -force $hwh_file [file join $out_dir $out_name.hwh]

open_run impl_1
report_timing_summary -file [file join $out_dir timing_summary.rpt]
report_utilization    -file [file join $out_dir utilization.rpt]
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
close_design

info_msg "Done.\n    [file join $out_dir $out_name.bit]\n    [file join $out_dir $out_name.hwh]\n    setup WNS = $wns ns"
if {$wns ne "" && $wns < 0} {
    puts "WARNING: \[build_bitstream\] Timing not met (WNS = $wns ns); see timing_summary.rpt."
}
