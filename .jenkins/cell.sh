#!/bin/bash
# Build, test or clean one Jenkins matrix cell.
#   usage: cell.sh build|test|clean|print
# Configured by the environment the Jenkinsfile sets:
#   CELL             unique cell name, e.g. cpu_GCC14_OPENMP_SIMD_NATIVE
#   MODULES          modules to load after cmake and boost, in order
#   CMAKE_FLAGS      cell-specific configure flags
#   SLURM_FLAGS      allocation flags for the test run
#   CTEST_FLAGS      e.g. "-j 4" or "-L MPI"
#   OMP_NUM_THREADS  threads for the test run
#   LAUNCHER         srun (default) or salloc (MPI: ctest launches the ranks)
#   LD_LIBRARY_PATH_PREPEND  optional, applied after the modules are loaded
# CC, CXX and FC are honoured if set.
#
set -eo pipefail # no -u: the Lmod module function reads unset variables

action="${1:?usage: cell.sh build|test|clean|print}"
: "${CELL:?}"

modules="cmake/3.30.8 boost/1.85.0 ${MODULES}"
tag="${BUILD_TAG:-manual_${USER}}"
build_dir="build_${CELL}_${tag}"
install_dir="install_${CELL}_${tag}"
test_dir="/scratch/gpfs/TROMP/specfempp/jenkins/test_${CELL}_${tag}"

# The *_FLAGS variables hold several words each and are split on purpose.
# shellcheck disable=SC2206
configure_command=(cmake3 -S . -B "${build_dir}"
  -D CMAKE_BUILD_TYPE=Release
  -D CMAKE_INSTALL_PREFIX="${install_dir}/bin"
  -D SPECFEM_BUILD_TESTS=ON
  -D SPECFEM_BUILD_BENCHMARKS=OFF
  -D SPECFEMPP_TEST_DIR="${test_dir}"
  ${CMAKE_FLAGS})


# shellcheck disable=SC2206,SC2016
test_command=("${LAUNCHER:-srun}" --account rse ${SLURM_FLAGS}
  bash -c 'hostname; echo "nodes: ${SLURM_JOB_NODELIST}"; exec ctest "$@"'
  ctest ${CTEST_FLAGS} --output-on-failure --no-tests=error
  --output-junit junit.xml)

load_modules() {
  module purge
  for name in ${modules}; do
    module load "${name}"
  done
  if [[ -n "${LD_LIBRARY_PATH_PREPEND}" ]]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH_PREPEND}:${LD_LIBRARY_PATH}"
  fi
}

case "${action}" in
  build)
    load_modules
    "${configure_command[@]}"
    cmake3 --build "${build_dir}"
    ;;
  test)
    load_modules
    export OMP_PROC_BIND=spread OMP_PLACES=threads
    mkdir -p junit
    junit_dir="${PWD}/junit"
    cd "${test_dir}"
    # ctest writes junit.xml only on exit, so keep it even when tests fail.
    status=0
    "${test_command[@]}" || status=$?
    cp junit.xml "${junit_dir}/${CELL}.xml" || true
    exit "${status}"
    ;;
  clean)
    rm -rf "${build_dir}" "${install_dir}" "${test_dir}"
    ;;
  print)
    echo "modules:   ${modules}"
    echo "configure: ${configure_command[*]}"
    echo "test:      OMP_NUM_THREADS=${OMP_NUM_THREADS} ${test_command[*]}"
    ;;
  *)
    echo "unknown action '${action}'" >&2
    exit 2
    ;;
esac
