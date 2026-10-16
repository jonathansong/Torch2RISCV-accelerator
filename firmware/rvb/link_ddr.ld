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
    /* (first: an input section goes to the first rule it matches) the generated code and its
     * runtime (rv_dispatch.py --code compiled), whole objects into DDR */
    .rvb_gen : { KEEP(*(.rvb_gen)) *rvb_gen_ll.o(.text .text.* .rodata .rodata.* .srodata .srodata.*)
                 *sa_rv_rt.o(.text .text.* .rodata .rodata.* .srodata .srodata.* .data .data.* .sdata .sdata.*) } > ddr
    .text : { KEEP(*(.text.start)) *(.text .text.*) } > bram
    .rodata : { *(.rodata .rodata.* .srodata .srodata.*) } > bram
    .data   : { *(.data .data.* .sdata .sdata.*) } > bram
    .bss (NOLOAD) : { __bss_start = .; *(.bss .bss.* .sbss .sbss.* COMMON) __bss_end = .; } > bram
    /DISCARD/ : { *(.eh_frame .eh_frame_hdr) }        /* (no unwinding; it would tie BRAM to the DDR code) */
    __stack_top = ORIGIN(bram) + LENGTH(bram);
}
