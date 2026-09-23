/* PicoRV32 program BRAM: 8 KB at 0xC0000000 (reset vector = start).
 * The top 256 bytes hold the mailbox (see mailbox.h). */
MEMORY
{
    bram (rwx) : ORIGIN = 0xC0000000, LENGTH = 0x1F00
}

ENTRY(_start)

SECTIONS
{
    .text : {
        KEEP(*(.text.start))
        *(.text .text.*)
    } > bram

    .rodata : { *(.rodata .rodata.* .srodata .srodata.*) } > bram
    .data   : { *(.data .data.* .sdata .sdata.*) } > bram

    .bss (NOLOAD) : {
        __bss_start = .;
        *(.bss .bss.* .sbss .sbss.* COMMON)
        __bss_end = .;
    } > bram

    __stack_top = ORIGIN(bram) + LENGTH(bram);
}
