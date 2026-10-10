#!/bin/sh
# Board memory layout, read-only (docs/kv260_upgrade_plan.md K5-M, step M1):
# what the accelerator pool's position and the DTB's install path depend on.
# Changes nothing.
#
#   on the board: sudo sh board_mem_inspect.sh [out]      (default mem_inspect.txt)
#   back:         scp ubuntu@<board>:mem_inspect.txt build/
out=${1:-mem_inspect.txt}
home=$(getent passwd "${SUDO_USER:-$USER}" | cut -d: -f6)
hex() { od -An -tx1 -v "$1" | tr -d ' \n'; echo; }

{
echo "== system"; uname -a; cat /etc/os-release 2>/dev/null | grep -E "^(NAME|VERSION)="
echo "== /proc/cmdline"; cat /proc/cmdline
echo "== DT /chosen"
for f in bootargs linux,initrd-start linux,initrd-end; do
    p=/sys/firmware/devicetree/base/chosen/$f
    [ -f "$p" ] && { printf '%s: ' "$f"; if [ "$f" = bootargs ]; then tr '\0' ' ' < "$p"; echo; else hex "$p"; fi; }
done
echo "== /proc/meminfo"; grep -E "MemTotal|MemFree|MemAvailable|Cma" /proc/meminfo
echo "== /proc/iomem (System RAM, reserved, kernel)"; cat /proc/iomem
echo "== DT /reserved-memory"
r=/sys/firmware/devicetree/base/reserved-memory
if [ -d "$r" ]; then
    for f in "#address-cells" "#size-cells"; do [ -f "$r/$f" ] && { printf '%s: ' "$f"; hex "$r/$f"; }; done
    for d in "$r"/*/; do
        echo "-- $(basename "$d")"
        for f in compatible reg size alignment alloc-ranges reusable no-map linux,cma-default; do
            [ -f "$d$f" ] || continue
            case $f in compatible) printf '  %s: ' "$f"; tr '\0' ' ' < "$d$f"; echo ;;
                        reusable|no-map|linux,cma-default) echo "  $f" ;;
                        *) printf '  %s: ' "$f"; hex "$d$f" ;; esac
        done
    done
else
    echo "(none)"
fi
echo "== DT memory nodes"
for d in /sys/firmware/devicetree/base/memory*; do [ -f "$d/reg" ] && { printf '%s reg: ' "$(basename "$d")"; hex "$d/reg"; }; done
echo "== DT model / compatible"
tr '\0' ' ' < /sys/firmware/devicetree/base/model; echo
tr '\0' ' ' < /sys/firmware/devicetree/base/compatible; echo
echo "== dmesg (memory, CMA, reserved)"
dmesg | grep -iE "reserved mem|cma|Memory:|OF: reserved|Machine model|Kernel command line|initrd|fdt" | head -60
echo "== /boot"; ls -la /boot 2>/dev/null; echo "-- /boot/firmware"; ls -la /boot/firmware 2>/dev/null
echo "-- dtb files"; find /boot /lib/firmware -maxdepth 4 \( -name "*.dtb" -o -name "*.dtbo" \) 2>/dev/null | head -40
echo "== flash-kernel"; cat /etc/default/flash-kernel 2>/dev/null; ls /etc/flash-kernel 2>/dev/null
dpkg -l 2>/dev/null | grep -E "flash-kernel|u-boot|xlnx-firmware|xlnx-config|linux-image" | awk '{print $2, $3}'
echo "== boot scripts"
for f in /boot/firmware/boot.scr* /boot/boot.scr* /boot/firmware/*.scr; do
    [ -f "$f" ] && { echo "-- $f"; strings "$f" | head -150; }
done
echo "== fw_printenv"; if command -v fw_printenv >/dev/null; then fw_printenv 2>&1 | head -120; else echo "(no fw_printenv)"; fi
echo "== xmutil / xlnx-config"; command -v xmutil xlnx-config 2>/dev/null
echo "== tools"; for t in dtc fdtput fdtget mkimage; do printf '%s: ' "$t"; command -v "$t" || echo missing; done
echo "== u-dma-buf"
ls -la "$home/udmabuf" 2>/dev/null | head; modinfo "$home/udmabuf/u-dma-buf.ko" 2>/dev/null | grep -E "^(version|vermagic|parm)"
lsmod | grep -i dma; for d in /sys/class/u-dma-buf/*; do [ -d "$d" ] && echo "$d: phys $(cat "$d/phys_addr") size $(cat "$d/size")"; done
echo "== loaded modules (dma / xrt / zocl)"; lsmod | grep -E "zocl|xrt|dma"
} > "$out" 2>&1
echo "written: $out"
