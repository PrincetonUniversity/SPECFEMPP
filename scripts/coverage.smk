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

# This file lives in scripts/; the repo root is its parent.
REPO_ROOT = Path(workflow.basedir).parent.resolve()


workdir: str(REPO_ROOT)


BUILD = "build/coverage"
PROFRAW_DIR = f"{BUILD}/profraw"
PROFDATA = f"{BUILD}/coverage.profdata"
BUILD_STAMP = f"{BUILD}/.coverage_build.stamp"
TEST_STAMP = f"{BUILD}/.coverage_tests.stamp"

# Parallel test jobs: at least 4, more if the machine has more cores.
# Override with CTEST_JOBS=N. The %p (PID) in LLVM_PROFILE_FILE keeps each
# parallel test process's raw profile separate, so parallelism is safe.
CTEST_JOBS = int(os.environ.get("CTEST_JOBS", max(4, os.cpu_count() or 4)))

# Library code only: drop dependencies, tests, and system/SDK headers.
IGNORE_REGEX = r"(/_deps/|/tests/|/usr/|/Applications/|/Library/Developer/)"

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
        "cmake --preset coverage"


rule build:
    # Ninja parallelizes the build internally across all available cores.
    input:
        f"{BUILD}/CMakeCache.txt",
    output:
        touch(BUILD_STAMP),
    shell:
        "cmake --build {BUILD}"


rule test:
    input:
        BUILD_STAMP,
    output:
        touch(TEST_STAMP),
    threads: CTEST_JOBS
    shell:
        # %p = PID, so parallel test processes never clobber each other's profiles.
        """
        rm -rf {PROFRAW_DIR}
        mkdir -p {PROFRAW_DIR}
        LLVM_PROFILE_FILE="$(pwd)/{PROFRAW_DIR}/cov-%p.profraw" \
            ctest --test-dir {BUILD}/tests/unit-tests --output-on-failure -j {CTEST_JOBS}
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
        html=directory(f"{BUILD}/coverage-html"),
    shell:
        # Test executables are copied flat into <build>/tests/unit-tests
        # (extension-less); discovery/data files carry extensions and are skipped.
        # llvm-cov takes one positional binary; the rest are passed via -object.
        """
        objects=()
        while IFS= read -r -d '' f; do objects+=("$f"); done \
            < <(find {BUILD}/tests/unit-tests -maxdepth 1 -type f ! -name '*.*' -print0)
        if [ ${{#objects[@]}} -eq 0 ]; then
            echo "ERROR: no instrumented test binaries found" >&2; exit 1
        fi
        cov_objects=("${{objects[0]}}")
        for b in "${{objects[@]:1}}"; do cov_objects+=(-object "$b"); done

        {LLVM_COV} report "${{cov_objects[@]}}" \
            -instr-profile={input.profdata} \
            -ignore-filename-regex='{IGNORE_REGEX}' | tee {output.summary}
        {LLVM_COV} show "${{cov_objects[@]}}" \
            -instr-profile={input.profdata} \
            -ignore-filename-regex='{IGNORE_REGEX}' \
            -format=html -output-dir={output.html}
        {LLVM_COV} export "${{cov_objects[@]}}" \
            -instr-profile={input.profdata} \
            -ignore-filename-regex='{IGNORE_REGEX}' \
            -format=lcov > {output.lcov}
        """


rule clean:
    shell:
        "rm -rf build/coverage"
