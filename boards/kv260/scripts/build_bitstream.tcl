# ----------------------------------------------------------------------
# One-shot, non-interactive build of the PicoRV32 + accelerator overlay for
# the Kria KV260 (docs/kv260_upgrade_plan.md K1a / K1b / K1c).
#
#   ./boards/kv260/scripts/build_bitstream.sh [-jobs N] [-sa_d 8|16] [-sa_mhz MHZ] [-dma_w 64|128] [-vefp_nb N] [-gemv 0|1] [-nports 1|2|4] [-m1_hp 2|3] [-m3_port hpc0|hp0] [-tag NAME] [-proj_dir DIR] [-bd_only] [-synth_only]
#
#   -sa_d    array size D (sa_unit parameter); default 8: K1a, the Z1 L2 accelerator
#   -sa_mhz  pl_clk0, the one PL clock; default 50 (K1a; K1b: 100)
#   -dma_w   the accelerator's DMA data width (S_AXI_HP1_FPD); default 64, K2a: 128 (config suffix _w128)
#   -vefp_nb batched special functions in the fp VE (sa_unit VEFP_NB, plan P6); default 8
#            (config suffix _p6), 0: off (the configurations before P6)
#   -gemv    1: the GEMV unit, LD mode GEMV (sa_unit GEMV, K2b; D = 16, DMA 128 bits;
#            config suffix _gemv); default 0
#   -nports  the accelerator's DMA ports (sa_unit NPORTS, K2b G3); 2: m1_axi on a second HP
#            port (config suffix _np2); 4 (G5): m1 HP3, m2 HP2, m3 per -m3_port (suffix
#            _np4); default 1
#   -m1_hp   the HP port of m1_axi: 3 (default, own DDR controller port) or 2 (shares S4
#            with HP1; config suffix _hp2); -nports 4 needs 3
#   -m3_port -nports 4: the PS port of m3_axi: hpc0 (default; PicoRV32 stays on HP0) or
#            hp0 (PicoRV32 moves to HPC0; config suffix _m3hp0)
#   -tag     appended to the configuration's name (_NAME): a new build of the same
#            parameters next to an accepted one
#   -bd_only build and validate the block design and write the address map,
#            no synthesis (a quick check of kv260_bd.tcl)
#   -synth_only  stop after synthesis (utilization of the synthesized design)
#   -jobs    threads of the main Vivado process (everything runs in it, see below)
#
# Memory: everything runs in this one Vivado process (in-project
# synth_design / opt / place / route, IP output products in global mode)
# instead of out-of-context runs plus synth_1 / impl_1 child processes. Each
# child loads the UltraScale+ device data again (~3.3 GB per IP run) while
# the main process holds its own copy, which on a 15 GB machine drove the
# session's memory pressure past systemd-oomd's limit.
#
# Flow: fresh project (part xck26-sfvc784-2LV-c, board kv260_som) -> IP
# repository rtl/ip (picorv32_axi) -> kv260_bd.tcl -> wrapper ->
# synth -> impl -> write_bitstream -> boards/kv260/build/output/d<D>_<MHz>mhz/picorv32.{bit,hwh}
# (one directory per configuration, e.g. d8_50mhz for K1a, d8_100mhz for K1b;
# the Vivado project likewise: build/picorv32_kv260_d<D>_<MHz>mhz)
# (+ timing / utilization reports, the address map). PYNQ needs the .bit and
# .hwh to share a basename. No PL I/O, so no pin constraints.
# ----------------------------------------------------------------------

set script_folder [file dirname [file normalize [info script]]]
set kv_root       [file dirname $script_folder]
set repo_root     [file dirname [file dirname $kv_root]]

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
set dma_w    64
set vefp_nb  8
set gemv     0
set nports   1
set m1_hp    3
set m3_port  hpc0
set tag      ""
set bd_only  0
set synth_only 0
set proj_dir ""

proc die {msg} { puts "\nERROR: \[kv260_build\] $msg\n"; exit 1 }
proc info_msg {msg} { puts "\nINFO: \[kv260_build\] $msg" }

for {set i 0} {$i < [llength $argv]} {incr i} {
    set arg [lindex $argv $i]
    switch -- $arg {
        -jobs     { set jobs [lindex $argv [incr i]] }
        -sa_d     { set sa_d [lindex $argv [incr i]] }
        -sa_mhz   { set sa_mhz [lindex $argv [incr i]] }
        -dma_w    { set dma_w [lindex $argv [incr i]] }
        -vefp_nb  { set vefp_nb [lindex $argv [incr i]] }
        -gemv     { set gemv [lindex $argv [incr i]] }
        -nports   { set nports [lindex $argv [incr i]] }
        -m1_hp    { set m1_hp [lindex $argv [incr i]] }
        -m3_port  { set m3_port [lindex $argv [incr i]] }
        -tag      { set tag [lindex $argv [incr i]] }
        -proj_dir { set proj_dir [file normalize [lindex $argv [incr i]]] }
        -bd_only  { set bd_only 1 }
        -synth_only { set synth_only 1 }
        default   { die "Unknown argument '$arg'. Valid: -jobs N, -sa_d 8|16, -sa_mhz MHZ, -dma_w 64|128, -vefp_nb N, -gemv 0|1, -nports 1|2|4, -m1_hp 2|3, -m3_port hpc0|hp0, -tag NAME, -proj_dir DIR, -bd_only, -synth_only" }
    }
}
if {![string is integer -strict $jobs] || $jobs < 1} { die "-jobs expects a positive integer, got '$jobs'" }
if {$sa_d != 8 && $sa_d != 16} { die "-sa_d expects 8 or 16, got '$sa_d'" }
if {$dma_w != 64 && $dma_w != 128} { die "-dma_w expects 64 or 128, got '$dma_w'" }
if {$vefp_nb != 0 && ($vefp_nb < 6 || $vefp_nb > 16)} { die "-vefp_nb expects 0 or 6..16, got '$vefp_nb'" }
if {$gemv != 0 && $gemv != 1} { die "-gemv expects 0 or 1, got '$gemv'" }
if {$gemv && ($sa_d != 16 || $dma_w != 128)} { die "-gemv 1 needs -sa_d 16 -dma_w 128" }
if {$nports != 1 && $nports != 2 && $nports != 4} { die "-nports expects 1, 2 or 4, got '$nports'" }
if {$m1_hp != 2 && $m1_hp != 3} { die "-m1_hp expects 2 or 3, got '$m1_hp'" }
if {$nports == 4 && $m1_hp != 3} { die "-nports 4 uses HP2 for m2: -m1_hp must be 3" }
if {$m3_port ne "hpc0" && $m3_port ne "hp0"} { die "-m3_port expects hpc0 or hp0, got '$m3_port'" }
# the ports' PS slave interfaces (kv260_bd.tcl): PicoRV32, then m0, m1, ...
set pico_port HP0
set dma_ports [list HP1]
if {$nports >= 2} { lappend dma_ports HP$m1_hp }
if {$nports == 4} {
    lappend dma_ports HP2 [string toupper $m3_port]
    if {$m3_port eq "hp0"} { set pico_port HPC0 }
}
set port_desc "PicoRV32 $pico_port, DMA [join $dma_ports { }]"
if {![string is integer -strict $sa_mhz] || $sa_mhz < 10 || $sa_mhz > 300} { die "-sa_mhz expects 10..300, got '$sa_mhz'" }
set_param general.maxThreads [expr {$jobs > 8 ? 8 : $jobs}]
if {[string first $vivado_ver [version -short]] == -1} {
    die "the scripts are written for Vivado $vivado_ver, this is Vivado [version -short]"
}
info_msg "KV260: D = $sa_d, pl_clk0 = $sa_mhz MHz, DMA $dma_w bits, VEFP_NB $vefp_nb, GEMV $gemv, NPORTS $nports ($port_desc)"

set cfg     d${sa_d}_${sa_mhz}mhz[expr {$dma_w == 128 ? "_w128" : ""}][expr {$vefp_nb > 0 ? "_p6" : ""}][expr {$gemv ? "_gemv" : ""}][expr {$nports > 1 ? "_np$nports" : ""}][expr {$nports > 1 && $m1_hp == 2 ? "_hp2" : ""}][expr {$nports == 4 && $m3_port eq "hp0" ? "_m3hp0" : ""}][expr {$tag ne "" ? "_$tag" : ""}]
set out_dir [file join $kv_root build output $cfg]
if {$proj_dir eq ""} { set proj_dir [file join $kv_root build ${proj_name}_$cfg] }
info_msg "configuration $cfg -> $out_dir"

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
set_property ip_repo_paths [list [file join $repo_root rtl ip]] [current_project]
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

# IP output products for global synthesis (no out-of-context runs)
set_property synth_checkpoint_mode None $bd_file
generate_target all $bd_file
set wrapper [make_wrapper -files $bd_file -top -force]
add_files -norecurse $wrapper
set_property top ${top_bd}_wrapper [current_fileset]
update_compile_order -fileset sources_1

proc mem_note {step} {
    info_msg "$step done: [clock format [clock seconds] -format %T]"
}

# ------------------------------------------------------------ synthesis
info_msg "Synthesis in this process ($jobs threads)"
if {[catch {synth_design -top ${top_bd}_wrapper -part $part} err]} { die "Synthesis failed: $err" }
mem_note synth_design
write_checkpoint -force [file join $out_dir post_synth.dcp]
report_utilization -file [file join $out_dir utilization_synth.rpt]
if {$synth_only} {
    info_msg "Done (-synth_only): [file join $out_dir utilization_synth.rpt]"
    exit 0
}

# ------------------------------------------------------- implementation
info_msg "Implementation in this process"
if {[catch {opt_design} err]}   { die "opt_design failed: $err" }
mem_note opt_design
if {[catch {place_design} err]} { die "place_design failed: $err" }
mem_note place_design
if {[catch {phys_opt_design} err]} { die "phys_opt_design failed: $err" }
if {[catch {route_design} err]} { die "route_design failed: $err" }
mem_note route_design
write_checkpoint -force [file join $out_dir post_route.dcp]

# -------------------------------------------------------------- outputs
set bit_file [file join $out_dir $out_name.bit]
if {[catch {write_bitstream -force $bit_file} err]} { die "write_bitstream failed: $err" }
set hwh_file [lindex [glob -nocomplain [file join $proj_dir *.gen sources_1 bd $top_bd hw_handoff $top_bd.hwh]] 0]
if {$hwh_file eq ""} { die "Hardware handoff ($top_bd.hwh) not found under $proj_dir" }
file copy -force $hwh_file [file join $out_dir $out_name.hwh]

report_timing_summary -file [file join $out_dir timing_summary.rpt]
report_utilization    -file [file join $out_dir utilization.rpt]
report_clocks         -file [file join $out_dir clocks.rpt]
check_timing          -file [file join $out_dir check_timing.rpt]
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
set whs [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -hold]]
close_design

# what this build is (read by the deploy scripts' VERSION)
set commit "unknown"
catch {set commit [string trim [exec git -C $repo_root rev-parse HEAD]]}
set dirty ""
catch {exec git -C $repo_root diff --quiet HEAD -- rtl boards/kv260} err opts
if {[dict get $opts -code] != 0} { set dirty " (dirty)" }
set fh [open [file join $out_dir build_info.txt] w]
puts $fh "config $cfg (D = $sa_d, pl_clk0 = $sa_mhz MHz, DMA $dma_w bits, VEFP_NB $vefp_nb, GEMV $gemv, NPORTS $nports, $port_desc, part $part)"
puts $fh "commit $commit$dirty"
puts $fh "built [clock format [clock seconds] -format {%Y-%m-%dT%H:%M:%S}] (Vivado [version -short], -jobs $jobs)"
puts $fh "setup WNS $wns ns, hold WHS $whs ns"
close $fh

info_msg "Done.\n    [file join $out_dir $out_name.bit]\n    [file join $out_dir $out_name.hwh]\n    setup WNS = $wns ns, hold WHS = $whs ns"
if {($wns ne "" && $wns < 0) || ($whs ne "" && $whs < 0)} {
    puts "WARNING: \[kv260_build\] Timing not met (WNS = $wns, WHS = $whs); see timing_summary.rpt."
}
