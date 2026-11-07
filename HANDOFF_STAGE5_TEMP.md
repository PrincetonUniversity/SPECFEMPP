# TEMPORARY: Stage 5 handoff for a run on another machine

Delete this file when the rerun is done, and before any merge or PR. Port notes normally live
in `~/Documents/metal`; this one is here only because the user asked for it on 2026-11-07.

## Task

Rerun the Stage 5 timing of the SPECFEM++ Metal port on a quiet Mac, to inform gate G4: Metal at
least 1.5x faster than tuned OpenMP + SIMD on a 3D case, **both measured on the same machine**.
G4 decides whether the port is kept, and only the user closes it.

Measure only. Do not tune chunk size, team size, events or copies.

## What already exists (first machine, not a G4 result)

Apple M3, 4P + 4E cores, 24 GB, macOS 26.7.2. All runs had other apps open, load about 3-6, at the
user's choice. Two-point warm ms/step:

| case | elements | Metal | CPU | CPU / Metal |
|---|---|---|---|---|
| 2D | 4,800 | 0.28 | 0.75 (4 threads) | 2.7x |
| 3D | 1,620 | 0.98 | 2.40 (8 threads) | 2.4x |
| 3D | 12,960 | 4.38 | 16.37 (8 threads) | 3.7x |
| 3D | 103,680 | 34.33 | 139.61 (8 threads) | 4.1x |

- Cold start costs about 9 s in 3D and 6 s in 2D.
- `KOKKOS_ACPP_COARSE_GRAINED_EVENTS=1` was 3x slower in 3D and about 6x in 2D. Unexplained.
- SIMD gave only 1.39x at one thread, across different compilers, so the CPU baseline may not be
  fully tuned.

**Do not mix these numbers with the new machine's.** Use them only to see whether the pattern
repeats.

## Git state

- Branch `sycl-metal`, HEAD `2e8c73009`. It is **local only, with no remote branch**, so it has to
  be transferred (for example with `git bundle`). Pushing is the user's decision.
- This note is untracked, so a bundle or clone will not include it. Copy it separately.
- The work folder `~/Documents/metal` is **not under git**. Copy what is listed below.

## Copy from `~/Documents/metal`

Keep `metal/` and `specfempp/` as siblings: the tools find the repo at `ROOT.parent / 'specfempp'`.

- `tools/`, `patches/acpp/`, `HANDOFF.md`, `metal_port_plan.md`, `TASK_B_performance_baseline.md`,
  `results/stage5_baseline/report.md` (the full record), and `results/fresh9/criteria_frozen.json`
  with `results/pre_stage5/expected_outputs.json` for the regression check
- `build/kt_runs/guard_bench_1/dim3__homogeneous_halfspace` and
  `build/kt_runs/guard_bench_1/dim2__homogeneous-medium-flat-topography`: about 20 MB, the inputs
  for the case generators

**Path trap:** files in `guard_bench_1` contain absolute `/Users/ccui/Documents/metal/...` paths.
The generators rewrite them using the current location, so the replacement silently matches
nothing if the path differs, and the configs keep pointing at the old location. Use the same
path, or rewrite those paths first. Grep the generated `specfem_config.yaml` to confirm.

`tools/check_metal_port.sh` hard-codes `/Users/ccui/Documents/...` and uses `merge-cpu-ref` as its
CPU reference. It has not been run on any other machine.

## Build

Both builds must come from the same commit. Previous versions: LLVM 20.1.8 (`llvm@20`), libomp,
boost, cmake 3.31, metal-cpp and AdaptiveCpp `f4ff36814e3d` with 7 patches from `patches/acpp/`
(`tools/build_acpp.sh`). `VERSIONS.txt` lists only 4 patches and is out of date. The 9 Kokkos
patches are in `cmake/metal/kokkos-patches/` and apply automatically. The preset expects
`MacOSX26.5.sdk`. Build named targets; `io_tests` is not needed.

```sh
# Metal
export PATH=$HOME/Documents/metal/install/acpp/bin:$PATH MACOSX_DEPLOYMENT_TARGET=26.0
cmake --preset release-sycl-metal -B build/fix2204-metal
cmake --build build/fix2204-metal -j4 --target specfem

# CPU baseline, in a shell without the acpp PATH change
cmake --preset release-openmp -B build/s5-openmp-simd \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@20/bin/clang++ \
  "-DOpenMP_CXX_FLAGS=-fopenmp -I/opt/homebrew/opt/libomp/include" \
  -DOpenMP_CXX_LIB_NAMES=omp \
  -DOpenMP_omp_LIBRARY=/opt/homebrew/opt/libomp/lib/libomp.dylib \
  -DKokkos_ARCH_NATIVE=OFF -DKokkos_ARCH_ARMV84=ON -DSPECFEM_ENABLE_SIMD=ON \
  -DSPECFEM_ENABLE_DOUBLE_PRECISION=OFF -DSPECFEM_BUILD_BENCHMARKS=OFF
cmake --build build/s5-openmp-simd -j4 --target specfem
```

`tools/make_stage5_sweep.py` also needs `build/merge-cpu-ref/bin/xmeshfem3D`. On the first machine
that was a float Serial, no-SIMD build at the #2221 merge. `make_stage5_sweep.py` and
`make_stage5_cases.py` default to the directory names above.

Before timing, check that the Metal build works on this machine. The first machine's correctness
evidence does not transfer. The solver log must show
`SYCL device: <chip> (AdaptiveCpp Metal backend)`.

## Recheck on a different chip

- **Scratch budget:** `BUDGET = 16368` and the scratch figures in the generators came from the M3,
  and the 2D case fits with only 108 B to spare. Recheck if the chip differs.
- **Thread settings:** sweep again for this core count. Test the performance-core count and all
  cores, with `OMP_PROC_BIND=false`. Binding did not help on the M3.
- **Memory:** keep device allocations well below the limit reported for this machine (14.3 GB on
  the 24 GB M3). Peak RSS was about 2.6 GB at 103,680 elements.

## Run

```sh
cd ~/Documents/metal
uptime; pmset -g batt; pmset -g therm      # mains power, no thermal warning, load below 1.5
python3 tools/make_stage5_cases.py build/kt_runs/s5_cases --omp-threads N
python3 tools/run_stage5_timing.py build/kt_runs/s5_cases/CASE.json build/kt_runs/s5_main/CASE --repetitions 7
python3 tools/stage5_step_rate.py SHORT/results.json LONG/results.json
python3 tools/make_stage5_sweep.py build/kt_runs/s5_sweep 1 2 4
```

- Run the short and long specs for 3D and 2D, on Metal and CPU. For coarse-grained events, copy a
  Metal spec and add `"KOKKOS_ACPP_COARSE_GRAINED_EVENTS": "1"` to its `env`.
- Keep the default `--load-limit 1.5`. If it cannot be reached, report the figure and ask the user.
  Do not start other heavy work during timing.
- The harness waits for load to fall between repetitions, so 8-thread runs can take far longer
  than their solver time (the largest specs took about 20 minutes).
- Sweep steps are 200 and 400, so those rates include a stability check every 10 steps. The main
  specs (1,000+ steps) cancel the checks.

## Traps

- The shell may be zsh, which does not word-split `$var`. A loop over `set -- $d` silently ran
  nothing. Name specs explicitly.
- Band codes in seismogram file names depend on `dt`: `MX*` at 0.16, `BX*` at 0.08 and 0.04. A
  further sweep factor may need another code in `seismograms()` in `make_stage5_sweep.py`.
- The solver summary goes to `OUTPUT_FILES/output.log`, not stdout.
- Do not run whole-disk `find` or similar during timing.
- OpenMP `abort_tests` can hang in `exit()`, and ctest logs a killed one as Passed. Check for strays.
- The first machine runs Cortex XDR, which has killed `io_tests` and the editor. Do not try to get
  around endpoint security on any machine.

## Still blocked, needs the user

- **Per-kernel timers and measured bandwidth:** `cmake/kokkos.cmake` forces
  `Kokkos_ENABLE_LIBDL=OFF` on Apple, which should stop `KOKKOS_TOOLS_LIBS` loading a tool (not
  tested). An instrumented build is proposed in the report but not approved. Use
  `KOKKOS_TOOLS_GLOBALFENCES=1` so Metal kernel times mean completed work.
- **Metal System Trace:** needs Xcode, which the first machine lacks. If this machine has it, one
  3D trace is in scope.
- **Not run yet:** 349,920 elements and the 2D sweep.

## When done

Record results with chip, core count, memory, macOS version and binary hashes in
`results/stage5_baseline/report.md`, as a separate section from the first machine. Bring them back
to `~/Documents/metal`, and delete this note.
