# Out-of-context synth + place + route of matmul_unit on the PYNQ-Z1 part.
# Usage: vivado -mode batch -source synth_ooc.tcl -tclargs <clock_ns> <rtl.v>...
set clock_ns [lindex $argv 0]
set rtl      [lrange $argv 1 end]

read_verilog $rtl
synth_design -top matmul_unit -part xc7z020clg400-1 -mode out_of_context
create_clock -name aclk -period $clock_ns [get_ports aclk]
opt_design
place_design
route_design

report_utilization    -file utilization.rpt
report_timing_summary -file timing_summary.rpt

set wns [get_property SLACK [get_timing_paths -max_paths 1 -setup]]
set fmax [format %.1f [expr {1000.0 / ($clock_ns - $wns)}]]
puts "RESULT: target [expr {1000.0 / $clock_ns}] MHz, WNS = $wns ns, est. fmax = $fmax MHz"
puts "RESULT: see [pwd]/utilization.rpt"
