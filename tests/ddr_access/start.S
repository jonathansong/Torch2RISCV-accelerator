    .section .text.start
    .global _start
_start:
    la    sp, __stack_top

    /* zero .bss */
    la    t0, __bss_start
    la    t1, __bss_end
1:  bgeu  t0, t1, 2f
    sw    zero, 0(t0)
    addi  t0, t0, 4
    j     1b
2:
    call  main

    /* ebreak -> PicoRV32 trap -> irq to the PS interrupt controller;
     * the core stays halted until the ARM asserts reset again. */
    ebreak
3:  j     3b
