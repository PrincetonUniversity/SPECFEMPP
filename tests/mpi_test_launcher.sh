#!/bin/sh
# Launcher wrapper for MPI gtest executables.
#
#   mpi_test_launcher.sh <mpiexec> <numproc-flag> <ranks> <binary> [args...]
#
# gtest_discover_tests() runs the test binary to enumerate its cases, and CMake hands it
# the same executor it puts in front of the test command: TEST_LAUNCHER and
# CROSSCOMPILING_EMULATOR both feed a single `test_executor` used for discovery and
# execution alike, so no target property can tell the two apart.
#
# That matters because listing is a local operation. Run under `mpiexec -n <ranks>`,
# every rank writes the listing to one merged stdout and gtest_discover_tests() turns
# each line into a test, so every case was registered once per rank -- 244 ctest entries
# for 58 real tests, each entry then paying its own mpiexec launch.
#
# The two invocations do differ in one place: their arguments. Discovery passes
# --gtest_list_tests, a real test run never does. So branch on that -- send the listing
# pass to a single process, and run everything else under MPI.
set -e

mpiexec=$1
numproc_flag=$2
ranks=$3
shift 3

for arg in "$@"; do
    if [ "$arg" = "--gtest_list_tests" ]; then
        exec "$@"
    fi
done

exec "$mpiexec" "$numproc_flag" "$ranks" "$@"
