# Out-of-context synth + place + route of sa_unit on the PYNQ-Z1 part.
# Usage: vivado -mode batch -source synth_ooc.tcl -tclargs <clock_ns> <srcdir> <generics...>
set clock_ns [lindex $argv 0]
set srcdir   [lindex $argv 1]
set generics [lrange $argv 2 end]

set_param general.maxThreads 4    ;# keep Vivado to 4 threads
read_verilog [glob $srcdir/*.v]
set_property include_dirs $srcdir [current_fileset]
synth_design -top sa_unit -part xc7z020clg400-1 -mode out_of_context \
    -include_dirs $srcdir -generic $generics
create_clock -name aclk -period $clock_ns [get_ports aclk]
opt_design
place_design
route_design

report_utilization    -file utilization.rpt
report_utilization    -hierarchical -hierarchical_depth 2 -file utilization_hier.rpt
report_timing_summary -file timing_summary.rpt

set wns [get_property SLACK [get_timing_paths -max_paths 1 -setup]]
puts "RESULT: target [expr {1000.0 / $clock_ns}] MHz, WNS = $wns ns, est. fmax = [format %.1f [expr {1000.0 / ($clock_ns - $wns)}]] MHz"
