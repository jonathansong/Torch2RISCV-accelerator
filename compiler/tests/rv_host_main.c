/* Host harness of the generated RISC-V code (compiler/tests/rv_equiv.py):
 * the export's LLVM IR compiled for the host with firmware/rv/sa_rv_rt.c
 * (SA_RV_HOST), called with the bindings and push constants of a dispatch;
 * prints one line per PCPI operation (and "#" notes: the LDPARAM reads).
 *
 *   rv_host <ddr image> <its physical base> <n bindings> <addresses...> <n constants> <values...> [<reads>]
 * <reads>: "address value" lines, the LDPARAM reads of the descriptor path in
 * order (values a dispatch stored before reading them back): a load of the
 * next listed address returns its value, any other load the image's.
 * The entry point's symbol: -DSA_RV_ENTRY=<name>. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "sa_rv_rt.h"

void SA_RV_ENTRY(const uint32_t *bindings, const uint32_t *push_constants);

static uint8_t *mem;
static uint32_t mem_base, mem_size;
static uint32_t rd_addr[65536], rd_val[65536], nrd, ird;

void sa_rv_host_pcpi(const char *op, int n, uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t e)
{
    uint32_t v[5] = {a, b, c, d, e};
    printf("%s", op);
    for (int i = 0; i < n; ++i)
        printf(" %x", v[i]);
    printf("\n");
}

void sa_rv_host_note(const char *what, uint32_t a, uint32_t b) { printf("# %s %x %x\n", what, a, b); }

uint32_t sa_rv_host_load32(uint32_t addr)
{
    if (ird < nrd && rd_addr[ird] == addr)
        return rd_val[ird++];
    if (addr < mem_base || addr - mem_base + 4 > mem_size) {
        fprintf(stderr, "load32 %#x outside the image\n", addr);
        exit(2);
    }
    const uint8_t *p = mem + (addr - mem_base);
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

int main(int argc, char **argv)
{
    if (argc < 4)
        return 2;
    FILE *f = fopen(argv[1], "rb");
    if (!f)
        return 2;
    fseek(f, 0, SEEK_END);
    mem_size = (uint32_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    mem = malloc(mem_size);
    if (fread(mem, 1, mem_size, f) != mem_size)
        return 2;
    fclose(f);
    mem_base = (uint32_t)strtoul(argv[2], 0, 0);
    int k = 3, nb = atoi(argv[k++]);
    uint32_t bind[32] = {0}, pc[64] = {0};
    for (int i = 0; i < nb && i < 32; ++i)
        bind[i] = (uint32_t)strtoul(argv[k++], 0, 0);
    int nc = atoi(argv[k++]);
    for (int i = 0; i < nc && i < 64; ++i)
        pc[i] = (uint32_t)strtoul(argv[k++], 0, 0);
    if (k < argc) {
        FILE *r = fopen(argv[k], "r");
        unsigned a, v;
        while (r && nrd < 65536 && fscanf(r, "%x %x", &a, &v) == 2)
            rd_addr[nrd] = a, rd_val[nrd++] = v;
        if (r)
            fclose(r);
    }
    SA_RV_ENTRY(bind, pc);
    return 0;
}
