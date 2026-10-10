#!/bin/sh
# Builds a DTB with a dedicated accelerator pool (docs/kv260_upgrade_plan.md
# K5-M, step M1) from the running device tree (/sys/firmware/fdt). Installs
# nothing: it writes accel.dtb, accel.dts and the diff against the running tree.
#
#   on the board: sh accel_mem_dtb.sh BASE SIZE [CMA_SIZE]     (needs dtc: apt install device-tree-compiler)
#     BASE, SIZE  the pool, hex bytes; inside the low 2 GB, below 0x7FF00000 (PMU firmware),
#                 clear of the kernel image, initrd and other reserved regions
#                 (board_mem_inspect.sh lists them); SIZE a multiple of 1 MiB
#     CMA_SIZE    new size of the global CMA (linux,cma node), hex; omitted: unchanged.
#                 A cma= on the kernel command line overrides the node: Ubuntu for
#                 Kria sets cma=1000M in its boot script and has no linux,cma node, so
#                 there the CMA size goes to /etc/default/flash-kernel instead
#                 (accel_mem_install.sh)
#   KV260 (Ubuntu 22.04, kernel 5.15.0-1080; board_mem_inspect.sh 2026-10-10): kernel up to
#   0x0320_0000, a reserved page range at 0x0FFF_2000, the initrd at 0x75DA_E000-0x78FF_FFFF
#   at boot (U-Boot puts it just below its own area at 0x7BF0_0000-), so
#     sh accel_mem_dtb.sh 0x10000000 0x50000000     (1280 MiB, 0x1000_0000-0x5FFF_FFFF)
#
# The pool is a shared-dma-pool marked reusable (a per-device CMA area): u-dma-buf's
# dma_alloc_coherent then takes any size. A no-map pool allocates in powers of two,
# so 1536 MiB would round up to 2 GiB and fail. The u-dma-buf node makes
# /dev/udmabuf0 of SIZE bytes, sync-mode 2 (write-combined), as soon as the module
# loads (no udmabuf0= parameter); load it at boot (modules-load.d) so it takes the
# whole pool before anything else can.
set -eu
[ $# -ge 2 ] || { sed -n '2,20p' "$0"; exit 2; }
BASE=$(printf '0x%x' "$1"); SIZE=$(printf '0x%x' "$2"); CMA=${3:-}
command -v dtc >/dev/null || { echo "dtc missing: sudo apt install device-tree-compiler"; exit 1; }

end=$((BASE + SIZE)); base_n=$((BASE))
[ $((SIZE % 0x100000)) -eq 0 ] || { echo "SIZE is not a multiple of 1 MiB"; exit 1; }
[ $((BASE % 0x100000)) -eq 0 ] || { echo "BASE is not 1 MiB aligned"; exit 1; }
[ "$end" -le $((0x7FF00000)) ] || { echo "pool ends at $(printf 0x%x $end), past 0x7FF00000"; exit 1; }

# overlaps with the regions in /proc/iomem other than plain System RAM
# (kernel code / data, reserved, ...): a warning to look at, not a proof
overlap=0
# the global CMA (dmesg "cma: Reserved N MiB at 0x...") is placed anew at each boot: not a conflict
cma_at=$(dmesg 2>/dev/null | sed -n 's/.*cma: Reserved \([0-9]*\) MiB at \(0x[0-9a-f]*\).*/\2 \1/p' | head -1)
cma_lo=$(( ${cma_at%% *} + 0 )) 2>/dev/null || cma_lo=-1
[ -n "$cma_at" ] && cma_hi=$((cma_lo + (${cma_at##* } << 20))) || { cma_lo=-1; cma_hi=-1; }
while IFS= read -r line; do
    range=${line%% :*}; what=${line#*: }
    lo=$((0x${range%-*})); hi=$((0x${range#*-} + 1))
    case $what in "System RAM") continue ;; esac
    [ "$lo" = "$cma_lo" ] && [ "$hi" = "$cma_hi" ] && { echo "(the global CMA at $(printf 0x%x $lo) moves at boot: ignored)"; continue; }
    if [ "$lo" -lt "$end" ] && [ "$hi" -gt "$base_n" ]; then
        echo "WARNING: pool overlaps $(printf '0x%08x-0x%08x' $lo $((hi - 1))) $what"; overlap=1
    fi
done <<EOF
$(sed -n 's/^ *\([0-9a-f]*-[0-9a-f]*\) : \(.*\)$/\1 : \2/p' /proc/iomem 2>/dev/null)
EOF
# the running tree's static reserved-memory regions (2 + 2 cells on the ZynqMP) and the initrd
be() { od -An -tx1 -v -j "$2" -N "$3" "$1" | tr -d ' \n'; }
chk() {   # name lo hi(exclusive)
    if [ "$2" -lt "$end" ] && [ "$3" -gt "$base_n" ]; then
        echo "WARNING: pool overlaps $(printf '0x%x-0x%x' "$2" $(($3 - 1))) $1"; overlap=1
    fi
}
for d in /sys/firmware/devicetree/base/reserved-memory/*/; do
    [ -f "${d}reg" ] || continue
    chk "reserved-memory/$(basename "$d")" $((0x$(be "${d}reg" 0 8))) $((0x$(be "${d}reg" 0 8) + 0x$(be "${d}reg" 8 8)))
done
c=/sys/firmware/devicetree/base/chosen
if [ -f "$c/linux,initrd-start" ]; then
    n=$(wc -c < "$c/linux,initrd-start")
    chk initrd $((0x$(be "$c/linux,initrd-start" 0 "$n"))) $((0x$(be "$c/linux,initrd-end" 0 "$n")))
fi
[ "$(id -u)" = 0 ] || echo "(not root: /proc/iomem shows zero addresses, so the overlap check above is empty; run with sudo)"

dtc -q -I dtb -O dts -o base.dts /sys/firmware/fdt
grep -q "reserved-memory {" base.dts || { echo "the running tree has no /reserved-memory node"; exit 1; }
cma_node=$(awk '/reserved-memory \{/{r=1} r && /linux,cma(@[0-9a-f]*)? \{/{gsub(/^[ \t]+|[ \t]*\{.*$/,""); print; exit}' base.dts)

hi() { printf '0x%x' $(($1 >> 32)); }
lo() { printf '0x%08x' $(($1 & 0xFFFFFFFF)); }
node=accel@$(printf '%x' "$BASE")
{
    cat base.dts
    cat <<EOF

/* K5-M step M1: the accelerator pool (accel_mem_dtb.sh $BASE $SIZE ${CMA:-}) */
/ {
	reserved-memory {
		$node {
			compatible = "shared-dma-pool";
			reusable;
			reg = <$(hi $BASE) $(lo $BASE) $(hi $SIZE) $(lo $SIZE)>;
		};
EOF
    if [ -n "$CMA" ]; then
        [ -n "$cma_node" ] || { echo "no linux,cma node in the running tree (CMA from cma= on the command line?)" >&2; exit 1; }
        printf '\t\t%s {\n\t\t\tsize = <%s %s>;\n\t\t};\n' "$cma_node" "$(hi "$CMA")" "$(lo "$CMA")"
    fi
    cat <<EOF
	};
	udmabuf-accel {
		compatible = "ikwzm,u-dma-buf";
		device-name = "udmabuf0";
		size = <$(hi $SIZE) $(lo $SIZE)>;
		memory-region = <&{/reserved-memory/$node}>;
		sync-mode = <2>;
		dma-mask = <32>;
	};
};
EOF
} > accel.dts
dtc -q -I dts -O dtb -o accel.dtb accel.dts
dtc -q -I dtb -O dts -o accel_check.dts accel.dtb
diff -u base.dts accel_check.dts > accel.diff || true
echo "pool $node: $BASE..$(printf 0x%x $((end - 1))) ($((SIZE >> 20)) MiB)${CMA:+, linux,cma ($cma_node) -> $((CMA >> 20)) MiB}"
echo "written: accel.dtb, accel.dts, accel.diff (vs the running tree)"
[ $overlap = 0 ] || echo "check the WARNINGs above before installing"
