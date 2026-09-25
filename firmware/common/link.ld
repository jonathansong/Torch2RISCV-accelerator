/* PicoRV32 program BRAM: 8 KB at 0xC0000000 (reset vector = start).
 * The top 512 bytes are reserved (see mailbox.h): 0x1E00-0x1EFF the
 * performance counter area, 0x1F00-0x1FFF the mailbox. Code, data and the
 * stack (top at 0x1E00) share the first 7.5 KB. */
MEMORY
{
    bram (rwx) : ORIGIN = 0xC0000000, LENGTH = 0x1E00
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
