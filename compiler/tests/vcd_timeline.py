#!/usr/bin/env python3
"""Clock-by-clock events of the first descriptor-list run in an xsim VCD of
tb_system (firmware/rvb/vcd.tcl, docs/e2e_walkthrough.md §9): commands issued
by the scheduler to LD / ST / EX / VE, the engines' done pulses, the end of the
list, and the DMA read beats, in cycles from the fetch unit's start.

    python3 compiler/tests/vcd_timeline.py <sim dir>/wave.vcd
"""
import sys, collections
path = sys.argv[1]
ids, vals, events = {}, {}, []
with open(path) as f:
    t = 0
    for line in f:
        p = line.split()
        if not p: continue
        if p[0] == '$var':
            ids[p[3]] = p[4]
        elif p[0].startswith('#'):
            t = int(p[0][1:])
        elif p[0][0] in '01xz' and len(p) == 1 and p[0][1:] in ids:
            events.append((t, ids[p[0][1:]], p[0][0]))
        elif p[0][0] == 'b' and len(p) == 2 and p[1] in ids:
            events.append((t, ids[p[1]], p[0][1:]))
# replay, sampling at each rising aclk edge
state = collections.defaultdict(lambda: '0')
byt = collections.defaultdict(list)
for t, n, v in events: byt[t].append((n, v))
cyc, start, out, counts = 0, None, [], collections.Counter()
prev_busy = '0'
for t in sorted(byt):
    clk_rise = False
    for n, v in byt[t]:
        if n == 'aclk' and v == '1' and state['aclk'] == '0': clk_rise = True
    if clk_rise:
        s = state
        if start is None and s['fetch_busy'] == '1': start = cyc
        if start is not None:
            c = cyc - start
            for eng in ('ld', 'st', 've', 'ex'):
                if s[eng + '_v'] == '1' and s[eng + '_r'] == '1': out.append((c, f'{eng.upper()} issued')); counts[eng] += 1
                if s[eng + '_done'] == '1': out.append((c, f'{eng.upper()} done'))
            if s['m0_axi_rvalid'] == '1' and s['m0_axi_rready'] == '1': counts['rbeat'] += 1
            if s['m0_axi_wvalid'] == '1': counts['wvalid_cycles'] += 1
            if s['fetch_reg_we'] == '1': out.append((c, 'SETREG/register write by fetch'))
            if prev_busy == '1' and s['fetch_busy'] == '0':
                out.append((c, 'fetch idle (END reached)'))
            if s['fetch_busy'] == '0' and s['sched_idle'] == '1' and c > 5:
                out.append((c, 'unit idle: list done')); break
            prev_busy = s['fetch_busy']
        cyc += 1
    for n, v in byt[t]: state[n] = v
for c, e in out: print(f'{c:7d}  {e}')
print('counts', dict(counts))
