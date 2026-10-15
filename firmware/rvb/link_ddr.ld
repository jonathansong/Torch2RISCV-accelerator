/* rvb with the generated code in DDR (RVB_CODE=ddr): everything else as
 * common/link.ld in the program BRAM, .rvb_gen at RVB_CODE_ADDR (after the
 * dispatch's DDR window; the PicoRV32 fetches it uncached over HP0). */
MEMORY
{
    bram (rwx) : ORIGIN = 0xC0000000, LENGTH = 0x1E00
    ddr  (rx)  : ORIGIN = RVB_CODE_ADDR, LENGTH = 0x10000
}
ENTRY(_start)
SECTIONS
{
    .text : { KEEP(*(.text.start)) *(.text .text.*) } > bram
    .rodata : { *(.rodata .rodata.* .srodata .srodata.*) } > bram
    .data   : { *(.data .data.* .sdata .sdata.*) } > bram
    .bss (NOLOAD) : { __bss_start = .; *(.bss .bss.* .sbss .sbss.* COMMON) __bss_end = .; } > bram
    .rvb_gen : { KEEP(*(.rvb_gen)) } > ddr          /* (called through a pointer: kept from --gc-sections) */
    __stack_top = ORIGIN(bram) + LENGTH(bram);
}
