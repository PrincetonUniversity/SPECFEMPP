# LLVM source-based code coverage for SPECFEM++ as a Snakemake workflow.
#
# Configures and builds the instrumented `coverage` CMake preset, runs the C++
# unit tests, merges the per-process raw profiles, and emits three artifacts
# under build/coverage/:
#   - coverage-summary.txt        terminal-style per-file summary
#   - coverage-html/index.html    browsable report (per-template-instantiation)
#   - coverage.lcov               lcov trace for Codecov
#
# Run (from anywhere; the workflow anchors itself at the repo root):
#   uv run snakemake -s scripts/coverage.smk --cores all
#   uv run snakemake -s scripts/coverage.smk --cores all clean   # remove build/coverage
#
# Tool overrides via env: LLVM_PROFDATA, LLVM_COV. The LLVM tools MUST match the
# Clang that built the binaries or `llvm-profdata merge` fails on a version
# mismatch; on macOS `xcrun` selects the version-matched Xcode toolchain tools.

import os
import platform
from pathlib import Path

import subprocess
import os

def get_performance_core_count():
    """Returns the number of performance cores on macOS, or None if it cannot be determined."""
    try:
        # Query macOS system control for the number of performance cores
        result = subprocess.check_output(['sysctl', '-n', 'hw.perflevel0.logicalcpu'])
        return int(result.strip())
    except (subprocess.SubprocessError, ValueError, FileNotFoundError):
        # Fallback to total cores if the command fails (e.g., older Intel Macs)
        return os.cpu_count() or 1

# This file lives in scripts/; the repo root is its parent.
REPO_ROOT = Path(workflow.basedir).parent.resolve()

workdir: str(REPO_ROOT)


BUILD = "build/coverage"
PROFRAW_DIR = f"{BUILD}/profraw"
PROFDATA = f"{BUILD}/coverage.profdata"
BUILD_STAMP = f"{BUILD}/.coverage_build.stamp"
TEST_STAMP = f"{BUILD}/.coverage_tests.stamp"

# Parallel jobs for BOTH the build (CMAKE_BUILD_PARALLEL_LEVEL) and the test run
# (CTEST_PARALLEL_LEVEL): at least 4, more if the machine has more cores. Override
# with CTEST_JOBS=N. We use the env vars rather than `-j N` because cmake/ctest
# >= 3.29 take an OPTIONAL -j argument, so a space-separated "-j 4" is silently
# dropped (the value is not consumed) and the tools fall back to serial/default.
CTEST_JOBS = int(os.environ.get("CTEST_JOBS", max(4, get_performance_core_count() or 4)))

# Library code only: drop dependencies, tests, and system/SDK headers.
IGNORE_REGEX = r"(/_deps/|/tests/|/usr/|/Applications/|/Library/Developer/)"

# Extra args appended verbatim to the `cmake --preset coverage` configure step.
# Empty by default (local runs are unaffected); CI sets this to wire a compiler
# launcher, e.g.
#   CMAKE_EXTRA_ARGS="-DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache"
CMAKE_EXTRA_ARGS = os.environ.get("CMAKE_EXTRA_ARGS", "")

if platform.system() == "Darwin":
    LLVM_PROFDATA = os.environ.get("LLVM_PROFDATA", "xcrun llvm-profdata")
    LLVM_COV = os.environ.get("LLVM_COV", "xcrun llvm-cov")
else:
    LLVM_PROFDATA = os.environ.get("LLVM_PROFDATA", "llvm-profdata")
    LLVM_COV = os.environ.get("LLVM_COV", "llvm-cov")


rule all:
    input:
        f"{BUILD}/coverage-summary.txt",
        f"{BUILD}/coverage.lcov",
        f"{BUILD}/coverage-html",
    localrule: True


rule configure:
    # Re-run configure when the build configuration changes.
    input:
        "CMakeLists.txt",
        "CMakePresets.json",
    output:
        f"{BUILD}/CMakeCache.txt",
    localrule: True
    shell:
        "cmake --preset coverage {CMAKE_EXTRA_ARGS}"


rule build:
    # CMAKE_BUILD_PARALLEL_LEVEL drives the build width. (A plain `-j N` is
    # unreliable here: cmake >= 3.29 makes -j's argument optional, so a
    # space-separated value is dropped and Ninja falls back to its native default.)
    input:
        f"{BUILD}/CMakeCache.txt",
    output:
        touch(BUILD_STAMP),
    shell:
        """
        echo Number of cores used to build: {CTEST_JOBS}
        export CMAKE_BUILD_PARALLEL_LEVEL={CTEST_JOBS}
        cmake --build {BUILD} -j{CTEST_JOBS}
        """


rule test:
    input:
        BUILD_STAMP,
    output:
        touch(TEST_STAMP),
    shell:
        # %m-%p names each profile by (binary signature, PID). ctest still runs
        # cases in parallel; %p keeps parallel writers separate, and the shared %m
        # prefix lets the report step group profiles per binary -- which is what
        # avoids the cross-binary hash-collision "mismatched data" in the lcov.
        # CTEST_PARALLEL_LEVEL sets the parallel width (a space-separated `-jN` is
        # dropped by ctest >= 3.29, whose -j argument is optional).
        """
        rm -rf {PROFRAW_DIR}
        mkdir -p {PROFRAW_DIR}
        echo Number of cores used to run the tests {CTEST_JOBS}
        LLVM_PROFILE_FILE="$(pwd)/{PROFRAW_DIR}/cov-%m-%p.profraw" \
            ctest --test-dir {BUILD}/tests/unit-tests --output-on-failure -j{CTEST_JOBS}
        """


rule merge:
    input:
        TEST_STAMP,
    output:
        PROFDATA,
    shell:
        "{LLVM_PROFDATA} merge -sparse {PROFRAW_DIR}/*.profraw -o {output}"


rule reports:
    input:
        profdata=PROFDATA,
        built=BUILD_STAMP,
    output:
        summary=f"{BUILD}/coverage-summary.txt",
        lcov=f"{BUILD}/coverage.lcov",
    shell:
        # Test executables are copied flat into <build>/tests/unit-tests
        # (extension-less, mode 755). The generator's Makefile (mode 644) and the
        # dotted discovery/data files are skipped via -perm -u+x + ! -name '*.*'.
        """
        objects=()
        while IFS= read -r -d '' f; do objects+=("$f"); done \
            < <(find {BUILD}/tests/unit-tests -maxdepth 1 -type f -perm -u+x ! -name '*.*' -print0)
        if [ ${{#objects[@]}} -eq 0 ]; then
            echo "ERROR: no instrumented test binaries found" >&2; exit 1
        fi

        # --- terminal summary: single merged profile.
        # llvm-cov takes one positional binary; the rest are passed via -object.
        # The 'mismatched data' warning here is expected -- mostly _deps/STL noise
        # from template instantiations shared across binaries -- and does NOT
        # affect the per-binary lcov produced below for Codecov.
        cov_objects=("${{objects[0]}}")
        for b in "${{objects[@]:1}}"; do cov_objects+=(-object "$b"); done
        {LLVM_COV} report "${{cov_objects[@]}}" \
            -instr-profile={input.profdata} \
            -ignore-filename-regex='{IGNORE_REGEX}' | tee {output.summary}

        # --- lcov for Codecov: prefer each binary's OWN profile (no hash-collision
        # drops), falling back to the merged profile for any binary we cannot
        # isolate. Per-binary steps are non-fatal -- one odd binary must never sink
        # the whole report. A binary's own profiles are its cov-<sig>-*.profraw
        # files; we recover <sig> with an instant --gtest_list_tests probe, run
        # from the test working directory (the cwd ctest uses, where the test data
        # is linked) so startup-time file access in a binary doesn't make the probe
        # abort. The binaries stay in the build tree, so address them absolutely.
        probe={BUILD}/probe
        probe_abs="$(pwd)/$probe"
        testdir="$(pwd)/{BUILD}/tests/run"
        : > {output.lcov}
        for b in "${{objects[@]}}"; do
            name=$(basename "$b")
            bin_abs="$(pwd)/$b"
            rm -rf "$probe"; mkdir -p "$probe"
            ( cd "$testdir" && LLVM_PROFILE_FILE="$probe_abs/p-%m-%p.profraw" \
                "$bin_abs" --gtest_list_tests ) >/dev/null 2>&1 || true
            one=""
            praw=$(ls "$probe"/p-*.profraw 2>/dev/null | head -1)
            if [ -n "$praw" ]; then
                sig=$(basename "$praw"); sig=${{sig#p-}}; sig=${{sig%.profraw}}; sig=${{sig%-*}}
                reals=( {PROFRAW_DIR}/cov-"$sig"-*.profraw )
                if [ -e "${{reals[0]}}" ] \
                   && {LLVM_PROFDATA} merge -sparse "${{reals[@]}}" -o "$probe/one.profdata" 2>/dev/null; then
                    one="$probe/one.profdata"
                fi
            fi
            if [ -z "$one" ]; then
                echo "WARN: $name -- using merged profile (could not isolate its own)" >&2
                one={input.profdata}
            fi
            {LLVM_COV} export "$b" -instr-profile="$one" \
                -ignore-filename-regex='{IGNORE_REGEX}' -format=lcov >> {output.lcov} \
                || echo "WARN: lcov export failed for $name; skipping" >&2
        done
        rm -rf "$probe"
        if [ ! -s {output.lcov} ]; then
            echo "ERROR: coverage.lcov is empty (no profiles exported)" >&2; exit 1
        fi
        """


rule report_html:
    # Browsable per-instantiation HTML report from the merged profile. Split out
    # from `reports` (and not part of the lcov path) because `llvm-cov show
    # -format=html` is the expensive step -- PR patch-coverage runs request only
    # the lcov + summary targets and skip this; full main/devel runs build `all`,
    # which includes it for the workflow artifact.
    input:
        profdata=PROFDATA,
        built=BUILD_STAMP,
    output:
        html=directory(f"{BUILD}/coverage-html"),
    shell:
        """
        objects=()
        while IFS= read -r -d '' f; do objects+=("$f"); done \
            < <(find {BUILD}/tests/unit-tests -maxdepth 1 -type f -perm -u+x ! -name '*.*' -print0)
        if [ ${{#objects[@]}} -eq 0 ]; then
            echo "ERROR: no instrumented test binaries found" >&2; exit 1
        fi
        cov_objects=("${{objects[0]}}")
        for b in "${{objects[@]:1}}"; do cov_objects+=(-object "$b"); done
        {LLVM_COV} show "${{cov_objects[@]}}" \
            -instr-profile={input.profdata} \
            -ignore-filename-regex='{IGNORE_REGEX}' \
            -format=html -output-dir={output.html}
        """


rule html_lcov:
    # Optional, on-demand: accurate browsable HTML built from the per-binary
    # coverage.lcov, so it matches the Codecov number (unlike the merged-profile
    # llvm-cov HTML, which drops hash-collided functions). Line/function based --
    # no per-instantiation region drill-down. genhtml merges the duplicate
    # per-file records from the concatenated trace automatically.
    # Requires `genhtml` from the lcov package
    # (macOS: brew install lcov; Debian/Ubuntu: apt-get install lcov):
    #   uv run snakemake -s scripts/coverage.smk --cores 1 html_lcov
    input:
        f"{BUILD}/coverage.lcov",
    output:
        directory(f"{BUILD}/coverage-html-lcov"),
    localrule: True
    shell:
        "genhtml {input} --output-directory {output} "
        "--title 'SPECFEM++ coverage' --legend"


rule clean:
    shell:
        "rm -rf build/coverage"
