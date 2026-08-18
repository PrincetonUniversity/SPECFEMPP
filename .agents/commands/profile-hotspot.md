You are a performance engineering assistant for SPECFEM++. Walk through the following workflow to identify, symbolicate, and propose optimizations for the top hotspot in a profiling trace.

> **Note:** This workflow requires macOS (xctrace, lldb). It is not available on Linux.

---

## Configuration

Derive all paths from the repository root (`REPO_ROOT=$(git rev-parse --show-toplevel)`):

- **Binary**: `$REPO_ROOT/bin/release-debug-info/specfem`
- **Binary on-disk `__TEXT` vmaddr**: `0x100000000` (Mach-O default for arm64/x86_64)
- **Benchmark run dir**: `$REPO_ROOT/benchmarks/build/release-debug-info/dim2/fluid-solid-interface/`
- **Benchmark invocation**: `specfem 2d -p <benchdir>/specfem_config.yaml`
- **Source root**: `$REPO_ROOT`

If `$ARGUMENTS` is provided, treat it as the path to an existing `.trace` file and skip recording.

---

## Step 1 — Record or locate a trace

If no trace path was given as an argument, check for a trace recorded within the last hour:

```bash
REPO_ROOT=$(git rev-parse --show-toplevel)
BENCHDIR="$REPO_ROOT/benchmarks/build/release-debug-info/dim2/fluid-solid-interface"
find "$BENCHDIR" \
  -maxdepth 1 -name "*.trace" -newer /tmp/.profile-hotspot-sentinel 2>/dev/null \
  | sort | tail -1
```

If nothing recent exists, record one now. **Important**: `cd` into the benchmark dir first and pass the full config path as argument — without `2d -p <config>` the binary exits immediately with a non-zero code and the trace contains no samples.

```bash
REPO_ROOT=$(git rev-parse --show-toplevel)
BENCHDIR="$REPO_ROOT/benchmarks/build/release-debug-info/dim2/fluid-solid-interface"
BINARY="$REPO_ROOT/bin/release-debug-info/specfem"
touch /tmp/.profile-hotspot-sentinel
cd "$BENCHDIR" && \
xcrun xctrace record \
  --template "Time Profiler" \
  --output . \
  --launch -- \
  "$BINARY" 2d -p "$BENCHDIR/specfem_config.yaml" 2>&1
```

Find the trace that was just created (newest `.trace` in the run dir):

```bash
ls -dt "$BENCHDIR"/*.trace | head -1
```

Store this path as `TRACE`. Verify the run succeeded by checking the TOC for `return-exit-status="0"` and a duration > 3 seconds. If the binary exited early or the duration is < 1s, the trace will have no useful samples.

---

## Step 2 — Export trace TOC

Use `--toc` (not `--xpath '/trace-toc'` — that returns "no content to export"):

```bash
xcrun xctrace export --input "$TRACE" --toc 2>/dev/null
```

Confirm the run shows `return-exit-status="0"` and `duration` > 3 seconds. The schemas of interest are `time-profile` (aggregated, weighted) and `time-sample` (raw). Use `time-profile` for hotspot analysis.

---

## Step 3 — Export the time-profile table to a file

Always export to a file first — the XML can be 1–2 MB and piping it to Python inline is unreliable. Use `run[@number="1"]` in the XPath (bare `/trace-toc/run/...` may not match):

```bash
xcrun xctrace export --input "$TRACE" \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="time-profile"]' \
  --output /tmp/tp.xml 2>&1
wc -c /tmp/tp.xml   # should be > 100 KB for a useful trace
```

---

## Step 4 — Parse, symbolicate, and analyze the profile

Use the hotspot report script at `scripts/xctrace_hotspot_report.py`. It:
- Parses the `time-profile` XML with proper id-ref resolution
- Distinguishes **true leaf** frames (the actual instruction pointer, even if in libc/libm/Kokkos) from the **nearest specfem caller**
- Computes **percentages** of total profiled time
- Uses **`lldb image lookup`** for symbolication (NOT `atos` — `atos` cannot inline-expand the heavily-templated Kokkos/SIMD/SPECFEM++ C++ and just echoes addresses back unchanged)
- Produces **caller→callee edge weights** so you can see *why* a function is hot
- Optionally emits **collapsed stack format** for flame graph tools (speedscope, flamegraph.pl)

### About `time-profile` XML structure

The `time-profile` XML uses **id-ref deduplication**: elements carry an `id` attribute and later occurrences use `ref` to point back. The Python script must build an id-map and resolve refs before reading attributes.

The specfem binary's **runtime load address** is in the `load-addr` attribute of `<binary name="specfem" ...>` elements inside the XML — no separate table lookup needed.

### About leaf-frame attribution

**Critical subtlety**: the old script treated the first specfem frame as the "leaf". This is wrong when the actual leaf is in a library called by specfem (e.g., `__bzero` from libc, Kokkos atomic ops, libm math functions). The script now tracks:
- **True self time**: attributed to the actual leaf frame (any binary)
- **Specfem self time**: attributed to the deepest specfem frame in each backtrace (the specfem function that "caused" the CPU time, even if the IP was in a callee library)
- **Inclusive time**: total time a specfem frame appears anywhere in a backtrace

Run it:

```bash
REPO_ROOT=$(git rev-parse --show-toplevel)
BINARY="$REPO_ROOT/bin/release-debug-info/specfem"
python3 "$REPO_ROOT/scripts/xctrace_hotspot_report.py" --binary "$BINARY"
```

With flame graph output:

```bash
python3 "$REPO_ROOT/scripts/xctrace_hotspot_report.py" --binary "$BINARY" --flamegraph /tmp/stacks.collapsed
```

### Interpreting the output

- **TRUE SELF TIME** — where the CPU actually was. If the top entry is `__bzero`, `_platform_memmove`, or `Kokkos::Impl::SharedAllocationTracker`, that tells you the *symptom* (memory zeroing, copies, atomic refcounts).
- **SPECFEM SELF TIME** — the deepest specfem function on the stack. This is the function *responsible* for the cost, even if the IP was in a library callee. This is usually the most actionable view.
- **INCLUSIVE TIME** — functions that appear anywhere on the stack. High inclusive + low self = the function is a caller/orchestrator, not the bottleneck itself.
- **CALLER → CALLEE EDGES** — shows which call sites drive cost. If `compute_stiffness` → `impl_load` is the top edge, the stiffness inner loop's load pattern is the issue.

---

## Step 5 — Read the relevant source

For each hot symbol, locate and read the source file identified by `lldb` in Step 4. Focus on:
1. The inner loop body where the hot address falls
2. Any `load_on_device` / `impl_load` calls inside callbacks passed to `gradient`, `divergence`, or `for_each_level`
3. The surrounding algorithmic context (what is being computed)

Key files commonly involved in SPECFEM++ stiffness computation:
- `core/specfem/compute/impl/compute_stiffness_interaction.tpp`
- `core/specfem/algorithms/gradient.hpp`
- `core/specfem/assembly/jacobian_matrix/dim2/impl_load.hpp`
- `core/specfem/assembly/jacobian_matrix/dim3/impl_load.hpp`
- `core/specfem/assembly/properties/*/impl_load.hpp`

---

## Step 6 — Plan optimizations

Analyze the hot code and propose concrete changes. Present the plan clearly before touching any file.

Structure the plan as:

### Hotspot summary
- Address, function, file:line
- What the code does
- Why it is expensive (memory access pattern, redundant computation, etc.)

### Proposed change
- Which file(s) to modify
- What the change is (show before/after pseudocode or diff snippet)
- Expected benefit (fewer loads, fewer index computations, etc.)
- Any risks or caveats

### Do not modify files yet — wait for explicit user approval.

---

## Known patterns from prior optimization work (use as guidance)

### Kokkos View copy-constructor overhead (NEW — 2026-02-25)
If the top self-time frames are in `Kokkos::Impl::SharedAllocationTracker::SharedAllocationTracker`,
`Kokkos::Impl::ReferenceCountedDataHandle::ReferenceCountedDataHandle`, or copy constructors of
`specfem::assembly::sources`, `specfem::medium_container::kernels::data_container`, or
`specfem::medium_container::properties::data_container` — the assembly object (which contains many
Kokkos Views) is being **copy-constructed inside the time loop**. Each Kokkos View copy does an
atomic reference-count increment, so copying a struct with N views costs N atomic ops.

**Fix**: find where the object is passed/captured by value in the hot path and change to
`const&` or `std::cref`. Look for lambda captures of assembly structs: `[assembly]` → `[&assembly]`.

### Double jacobian load in gradient callback
The `specfem::algorithms::gradient` function loads `point_jacobian_matrix` per GLL point internally. If the callback passed to `gradient` also calls `load_on_device(index, jacobian_matrix, ...)`, that triggers a second MDSpan `mapping(ispec, iz, ix)` computation plus 4–5 SIMD memory loads per point.

**Fix already applied**: `gradient.hpp` now detects via `std::is_invocable_v` whether the callback accepts a 3rd `point_jacobian_matrix` argument (`store_jacobian=true`). If yes, it loads once with `store_jacobian=true` and forwards — eliminating the second load. `compute_stiffness_interaction.tpp` callback updated accordingly.

If the hotspot is in a *different* callback that still re-loads jacobian, apply the same pattern.

### MDSpan `mapping(ispec, iz, ix)` cost (dim2 SIMD path)
In `assembly/jacobian_matrix/dim2/impl_load.hpp:34`, `mapping(ispec, iz, ix)` computes a flat index via the Kokkos MDSpan layout mapping. On CPU with SIMD, this is called once per chunk-SIMD-lane. On GPU it is once per thread. If called multiple times for the same index in the same scope, hoist `_index` out.

### Properties re-loaded on each divergence point
`compute_stiffness_interaction.tpp` also loads `point_property` inside the `divergence` callback. If properties are used identically in both gradient and divergence callbacks for the same element, there may be an opportunity to cache them in scratch memory.

### Scratch memory pressure
SPECFEM++ uses `ChunkElementFieldType`, `ChunkStressIntegrandType`, and `ElementQuadratureType` in team scratch. Avoid allocating additional large arrays in scratch inside hot loops; prefer register-level (stack) variables for per-point quantities.
