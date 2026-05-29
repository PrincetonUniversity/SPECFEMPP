#!/usr/bin/env python3
"""Parse xctrace time-profile XML, symbolicate with lldb, emit hotspot report.

Usage:  python3 scripts/xctrace_hotspot_report.py --binary <path> --xml /tmp/tp.xml

Requires: macOS with `lldb` in PATH (ships with Xcode command-line tools).
NOTE: atos is intentionally NOT used — it cannot inline-expand heavily-templated
Kokkos/SPECFEM++ C++ and echoes addresses back unchanged.
"""
import xml.etree.ElementTree as ET
import collections, subprocess, sys, argparse

# --------------- configuration ---------------
TEXT_VMADDR = 0x100000000   # on-disk __TEXT vmaddr (Mach-O default for arm64/x86_64)
XML_PATH = "/tmp/tp.xml"
TOP_N    = 25

# --------------- arg parsing ---------------
ap = argparse.ArgumentParser(
    description="Parse xctrace time-profile XML and emit hotspot report.")
ap.add_argument("--binary", required=True,
                help="path to the specfem binary (for lldb symbolication)")
ap.add_argument("--xml", default=XML_PATH, help="path to exported time-profile XML")
ap.add_argument("--flamegraph", default=None,
                help="if set, write collapsed-stack output for flame graphs to this path")
ap.add_argument("--top", type=int, default=TOP_N, help="number of top entries to show")
ap.add_argument("--thread", default=None,
                help="(not yet implemented) filter to a specific thread name")
args = ap.parse_args()

BINARY = args.binary

# --------------- XML parsing helpers ---------------
tree = ET.parse(args.xml)
root = tree.getroot()

id_map = {}
for el in root.iter():
    eid = el.get('id')
    if eid:
        id_map[eid] = el

def resolve(el):
    """Resolve id-ref deduplication."""
    ref = el.get('ref')
    return id_map[ref] if (ref and ref in id_map) else el

# --------------- discover specfem binary info ---------------
# Deduplicate by id(resolved_element) — UUID-only dedup fails when UUID is
# empty (common in CPU Counters traces), causing hundreds of spurious [info] lines.
specfem_uuids = set()
_seen_binary_ids = set()
load_addr = None
for b in root.iter('binary'):
    b = resolve(b)
    if 'specfem' not in b.get('name', '').lower():
        continue
    if id(b) in _seen_binary_ids:
        continue
    _seen_binary_ids.add(id(b))
    uuid = b.get('UUID', '')
    la_str = b.get('load-addr', '0')
    # Use UUID when present, else fall back to load-addr as surrogate key
    specfem_key = uuid if uuid else la_str
    specfem_uuids.add(specfem_key)
    if load_addr is None:
        load_addr = int(la_str, 16)
    print(f"[info] specfem binary: uuid={uuid}  load-addr={la_str}  path={b.get('path')}")

if load_addr is None:
    sys.exit("[error] specfem binary not found in XML — wrong trace or no samples?")

slide = load_addr - TEXT_VMADDR
print(f"[info] ASLR slide = {hex(slide)} ({slide})")
print()

def is_specfem(frame):
    """Return True if the resolved frame belongs to the specfem binary."""
    b_el = frame.find('binary')
    if b_el is None:
        return False
    b_el = resolve(b_el)
    uuid = b_el.get('UUID', '')
    la_str = b_el.get('load-addr', '')
    key = uuid if uuid else la_str
    return key in specfem_uuids

def frame_key(frame):
    """Return (addr, name) for a resolved frame."""
    return (frame.get('addr', ''), frame.get('name', '<unknown>'))

def frame_binary_name(frame):
    b_el = frame.find('binary')
    if b_el is None:
        return '???'
    b_el = resolve(b_el)
    return b_el.get('name', '???')

# --------------- aggregate profile data ---------------
# Counters
true_leaf_count     = collections.Counter()   # true leaf frame (any binary)
specfem_leaf_count  = collections.Counter()   # deepest specfem frame per backtrace
inclusive_count     = collections.Counter()   # all frames (specfem only)
caller_callee_edge = collections.Counter()   # (caller_key, callee_key) -> weight
total_weight_ms    = 0

# For flame graph output
collapsed_stacks = collections.Counter()  # "frameA;frameB;frameC" -> weight

for row in root.iter('row'):
    # Parse weight (nanoseconds -> milliseconds)
    w_el = row.find('weight')
    if w_el is None:
        continue
    w_el = resolve(w_el)
    try:
        weight = int(w_el.text) // 1_000_000
    except (ValueError, TypeError):
        weight = 1

    # Optional thread filter
    if args.thread:
        # thread info is not always in time-profile rows;
        # skip filter if not present
        pass  # TODO: thread filtering if schema provides it

    bt = row.find('backtrace')
    if bt is None:
        continue
    bt = resolve(bt)

    # Collect all frames in order (leaf-first in xctrace XML)
    frames = []
    for f in bt.iter('frame'):
        frames.append(resolve(f))

    if not frames:
        continue

    total_weight_ms += weight

    # --- True leaf (the actual IP, any binary) ---
    leaf_frame = frames[0]
    leaf_key = frame_key(leaf_frame)
    leaf_bin = frame_binary_name(leaf_frame)
    true_leaf_count[(leaf_key[0], leaf_key[1], leaf_bin)] += weight

    # --- Deepest specfem frame (first specfem frame, since frames are leaf-first) ---
    specfem_leaf_frame = None
    for f in frames:
        if is_specfem(f):
            specfem_leaf_frame = f
            break

    if specfem_leaf_frame is not None:
        specfem_leaf_count[frame_key(specfem_leaf_frame)] += weight

    # --- Inclusive time for all specfem frames ---
    seen = set()
    prev_specfem_key = None
    for f in frames:
        if is_specfem(f):
            key = frame_key(f)
            if key not in seen:
                inclusive_count[key] += weight
                seen.add(key)
            # Build caller->callee edges (frames go leaf->caller, so caller is later)
            if prev_specfem_key is not None and prev_specfem_key != key:
                caller_callee_edge[(key, prev_specfem_key)] += weight
            prev_specfem_key = key

    # --- Collapsed stack for flame graph (reversed: caller first) ---
    if args.flamegraph:
        stack_names = [frame_key(f)[1] for f in reversed(frames)]
        collapsed_stacks[";".join(stack_names)] += weight

# --------------- reporting helpers ---------------
def pct(ms):
    return f"{100.0 * ms / total_weight_ms:.1f}%" if total_weight_ms > 0 else "—"

# --------------- report: true leaf frames ---------------
print(f"Total profiled time: {total_weight_ms} ms\n")

print(f"=== TRUE SELF TIME — top {args.top} leaf frames (any binary) ===")
print(f"{'%':>6} {'ms':>7}  {'Binary':<20} {'Address':<18} Function")
print("-" * 100)
for (addr, name, bname), ms in true_leaf_count.most_common(args.top):
    print(f"{pct(ms):>6} {ms:>7}  {bname:<20} {addr:<18} {name[:55]}")

# --------------- report: specfem self time ---------------
print(f"\n=== SPECFEM SELF TIME — top {args.top} deepest-specfem-frame per sample ===")
print(f"{'%':>6} {'ms':>7}  {'Address':<18} Function")
print("-" * 90)
for (addr, name), ms in specfem_leaf_count.most_common(args.top):
    print(f"{pct(ms):>6} {ms:>7}  {addr:<18} {name[:60]}")

# --------------- report: inclusive time ---------------
print(f"\n=== INCLUSIVE TIME — top {args.top} specfem frames ===")
print(f"{'%':>6} {'ms':>7}  {'Address':<18} Function")
print("-" * 90)
for (addr, name), ms in inclusive_count.most_common(args.top):
    print(f"{pct(ms):>6} {ms:>7}  {addr:<18} {name[:60]}")

# --------------- report: caller->callee edges ---------------
print(f"\n=== HOT CALLER -> CALLEE EDGES (specfem only) — top {args.top} ===")
print(f"{'%':>6} {'ms':>7}  Caller -> Callee")
print("-" * 100)
for (caller_key, callee_key), ms in caller_callee_edge.most_common(args.top):
    c1 = caller_key[1][:42]
    c2 = callee_key[1][:42]
    print(f"{pct(ms):>6} {ms:>7}  {c1}  ->  {c2}")

# --------------- symbolication with lldb ---------------
# atos cannot inline-expand the heavily-templated Kokkos/SIMD/SPECFEM++ C++;
# it echoes addresses back unchanged. Use lldb image lookup instead.
addrs_to_resolve = set()
for (addr, name), ms in specfem_leaf_count.most_common(args.top):
    addrs_to_resolve.add(addr)
for (addr, name), ms in inclusive_count.most_common(min(10, args.top)):
    addrs_to_resolve.add(addr)

if addrs_to_resolve:
    print(f"\n=== SYMBOLICATION via lldb (slide={hex(slide)}) ===")
    addr_list = sorted(addrs_to_resolve)
    lldb_cmd_file = "/tmp/lldb_sym_cmds.txt"
    with open(lldb_cmd_file, 'w') as f:
        f.write(f"target create {BINARY}\n")
        for a in addr_list:
            file_a = int(a, 16) - slide
            f.write(f"image lookup -a {hex(file_a)}\n")
        f.write("quit\n")
    try:
        result = subprocess.run(
            ["lldb", "-b", "-s", lldb_cmd_file],
            capture_output=True, text=True, timeout=120
        )
        lines = result.stdout.split('\n')
        idx = 0
        for a in addr_list:
            print(f"  {a}:")
            while idx < len(lines):
                l = lines[idx].strip()
                if l.startswith('Address:') or l.startswith('Summary:') or l.startswith('LineEntry:'):
                    print(f"    {l}")
                elif l.startswith('(lldb)') and 'image lookup' in l:
                    idx += 1
                    break
                idx += 1
    except FileNotFoundError:
        print("  [warning] lldb not found — install Xcode command-line tools")
    except subprocess.TimeoutExpired:
        print("  [warning] lldb timed out")
    print(f"  [hint] To re-run manually: lldb -b -s {lldb_cmd_file} 2>&1 | grep -E 'Address:|Summary:|LineEntry:'")

# --------------- flame graph output ---------------
if args.flamegraph:
    with open(args.flamegraph, 'w') as f:
        for stack, w in collapsed_stacks.most_common():
            f.write(f"{stack} {w}\n")
    print(f"\n[info] Collapsed stacks written to {args.flamegraph}")
    print(f"       View with: cat {args.flamegraph} | flamegraph.pl > /tmp/fg.svg")
    print(f"       Or upload to https://www.speedscope.app/")
