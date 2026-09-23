# PYNQ-Z1 constraints for the picorv32 block design.
#
# Trimmed from Digilent's official "PYNQ-Z1 Rev C" Master XDC template and
# renamed to match the top-level ports actually created by
# scripts/pico_processor.tcl / scripts/pico_bit.tcl (led_o, pb_i,
# pmoda_gpio, pmodb_gpio, arduino_gpio_tri_io, arduino_direct_iic_*).
#
# Pin locations were cross-checked against constrs/PYNQ-Z2.xdc: for every
# signal below, PYNQ-Z1 and PYNQ-Z2 use the identical physical pin, so no
# re-derivation of pin numbers was needed -- only the port names differ.
#
# NOTE: Digilent's template labels PACKAGE_PIN P16 as ck_scl and P15 as
# ck_sda, which is the *opposite* of how constrs/PYNQ-Z2.xdc assigns those
# same two physical pins (P15=scl, P16=sda). The two vendors' own xdc files
# disagree on which pin is clock vs. data for this identical connector.
# The mapping below follows PYNQ-Z2.xdc's convention (P15=scl, P16=sda) to
# stay consistent with this project's existing IIC wiring -- verify against
# the PYNQ-Z1 schematic before relying on the Arduino I2C port on real
# hardware.

## LEDs
set_property -dict {PACKAGE_PIN R14 IOSTANDARD LVCMOS33} [get_ports {led_o[0]}]
set_property -dict {PACKAGE_PIN P14 IOSTANDARD LVCMOS33} [get_ports {led_o[1]}]
set_property -dict {PACKAGE_PIN N16 IOSTANDARD LVCMOS33} [get_ports {led_o[2]}]
set_property -dict {PACKAGE_PIN M14 IOSTANDARD LVCMOS33} [get_ports {led_o[3]}]

## Buttons
set_property -dict {PACKAGE_PIN D19 IOSTANDARD LVCMOS33} [get_ports {pb_i[0]}]
set_property -dict {PACKAGE_PIN D20 IOSTANDARD LVCMOS33} [get_ports {pb_i[1]}]
set_property -dict {PACKAGE_PIN L20 IOSTANDARD LVCMOS33} [get_ports {pb_i[2]}]
set_property -dict {PACKAGE_PIN L19 IOSTANDARD LVCMOS33} [get_ports {pb_i[3]}]

## Pmod Header JA -> pmoda_gpio
set_property -dict {PACKAGE_PIN Y18 IOSTANDARD LVCMOS33} [get_ports {pmoda_gpio[0]}]
set_property -dict {PACKAGE_PIN Y19 IOSTANDARD LVCMOS33} [get_ports {pmoda_gpio[1]}]
set_property -dict {PACKAGE_PIN Y16 IOSTANDARD LVCMOS33} [get_ports {pmoda_gpio[2]}]
set_property -dict {PACKAGE_PIN Y17 IOSTANDARD LVCMOS33} [get_ports {pmoda_gpio[3]}]
set_property -dict {PACKAGE_PIN U18 IOSTANDARD LVCMOS33} [get_ports {pmoda_gpio[4]}]
set_property -dict {PACKAGE_PIN U19 IOSTANDARD LVCMOS33} [get_ports {pmoda_gpio[5]}]
set_property -dict {PACKAGE_PIN W18 IOSTANDARD LVCMOS33} [get_ports {pmoda_gpio[6]}]
set_property -dict {PACKAGE_PIN W19 IOSTANDARD LVCMOS33} [get_ports {pmoda_gpio[7]}]
set_property PULLUP true [get_ports {pmoda_gpio[2]}]
set_property PULLUP true [get_ports {pmoda_gpio[3]}]
set_property PULLUP true [get_ports {pmoda_gpio[6]}]
set_property PULLUP true [get_ports {pmoda_gpio[7]}]

## Pmod Header JB -> pmodb_gpio
set_property -dict {PACKAGE_PIN W14 IOSTANDARD LVCMOS33} [get_ports {pmodb_gpio[0]}]
set_property -dict {PACKAGE_PIN Y14 IOSTANDARD LVCMOS33} [get_ports {pmodb_gpio[1]}]
set_property -dict {PACKAGE_PIN T11 IOSTANDARD LVCMOS33} [get_ports {pmodb_gpio[2]}]
set_property -dict {PACKAGE_PIN T10 IOSTANDARD LVCMOS33} [get_ports {pmodb_gpio[3]}]
set_property -dict {PACKAGE_PIN V16 IOSTANDARD LVCMOS33} [get_ports {pmodb_gpio[4]}]
set_property -dict {PACKAGE_PIN W16 IOSTANDARD LVCMOS33} [get_ports {pmodb_gpio[5]}]
set_property -dict {PACKAGE_PIN V12 IOSTANDARD LVCMOS33} [get_ports {pmodb_gpio[6]}]
set_property -dict {PACKAGE_PIN W13 IOSTANDARD LVCMOS33} [get_ports {pmodb_gpio[7]}]
set_property PULLUP true [get_ports {pmodb_gpio[2]}]
set_property PULLUP true [get_ports {pmodb_gpio[3]}]
set_property PULLUP true [get_ports {pmodb_gpio[6]}]
set_property PULLUP true [get_ports {pmodb_gpio[7]}]

## ChipKit/Arduino GPIO (Digital I/O Low + Outer Analog Header used as digital)
set_property -dict {PACKAGE_PIN T14 IOSTANDARD LVCMOS33} [get_ports {arduino_gpio_tri_io[0]}]
set_property -dict {PACKAGE_PIN U12 IOSTANDARD LVCMOS33} [get_ports {arduino_gpio_tri_io[1]}]
set_property -dict {PACKAGE_PIN U13 IOSTANDARD LVCMOS33} [get_ports {arduino_gpio_tri_io[2]}]
set_property -dict {PACKAGE_PIN V13 IOSTANDARD LVCMOS33} [get_ports {arduino_gpio_tri_io[3]}]
set_property -dict {PACKAGE_PIN V15 IOSTANDARD LVCMOS33} [get_ports {arduino_gpio_tri_io[4]}]
set_property -dict {PACKAGE_PIN T15 IOSTANDARD LVCMOS33} [get_ports {arduino_gpio_tri_io[5]}]
set_property -dict {PACKAGE_PIN R16 IOSTANDARD LVCMOS33} [get_ports {arduino_gpio_tri_io[6]}]
set_property -dict {PACKAGE_PIN U17 IOSTANDARD LVCMOS33} [get_ports {arduino_gpio_tri_io[7]}]
set_property -dict {PACKAGE_PIN V17 IOSTANDARD LVCMOS33} [get_ports {arduino_gpio_tri_io[8]}]
set_property -dict {PACKAGE_PIN V18 IOSTANDARD LVCMOS33} [get_ports {arduino_gpio_tri_io[9]}]
set_property -dict {PACKAGE_PIN T16 IOSTANDARD LVCMOS33} [get_ports {arduino_gpio_tri_io[10]}]
set_property -dict {PACKAGE_PIN R17 IOSTANDARD LVCMOS33} [get_ports {arduino_gpio_tri_io[11]}]
set_property -dict {PACKAGE_PIN P18 IOSTANDARD LVCMOS33} [get_ports {arduino_gpio_tri_io[12]}]
set_property -dict {PACKAGE_PIN N17 IOSTANDARD LVCMOS33} [get_ports {arduino_gpio_tri_io[13]}]
set_property -dict {PACKAGE_PIN Y11 IOSTANDARD LVCMOS33} [get_ports {arduino_gpio_tri_io[14]}]
set_property -dict {PACKAGE_PIN Y12 IOSTANDARD LVCMOS33} [get_ports {arduino_gpio_tri_io[15]}]
set_property -dict {PACKAGE_PIN W11 IOSTANDARD LVCMOS33} [get_ports {arduino_gpio_tri_io[16]}]
set_property -dict {PACKAGE_PIN V11 IOSTANDARD LVCMOS33} [get_ports {arduino_gpio_tri_io[17]}]
set_property -dict {PACKAGE_PIN T5  IOSTANDARD LVCMOS33} [get_ports {arduino_gpio_tri_io[18]}]
set_property -dict {PACKAGE_PIN U10 IOSTANDARD LVCMOS33} [get_ports {arduino_gpio_tri_io[19]}]

## ChipKit/Arduino direct I2C
set_property -dict {PACKAGE_PIN P15 IOSTANDARD LVCMOS33} [get_ports arduino_direct_iic_scl_io]
set_property -dict {PACKAGE_PIN P16 IOSTANDARD LVCMOS33} [get_ports arduino_direct_iic_sda_io]
set_property PULLUP true [get_ports arduino_direct_iic_scl_io]
set_property PULLUP true [get_ports arduino_direct_iic_sda_io]
