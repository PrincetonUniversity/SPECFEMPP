// What each axis value means.

def compiler_module = [GCC11: 'gcc/11', GCC14: 'gcc-toolset/14']
def simd_flags = [
    SIMD_NONE  : '-DSPECFEM_ENABLE_SIMD=OFF',
    SIMD_NATIVE: '-DSPECFEM_ENABLE_SIMD=ON -DKokkos_ARCH_SKX=ON -DKokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON'
]
def host_flags = [
    SERIAL: '-DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_ATOMICS_BYPASS=ON'
]
def mpi = [
    MPI_GCC416: [module: 'openmpi/gcc/4.1.6', slurm: '--ntasks=8 --nodes=2-8']
]
// Excluded nodes (reported to the sysadmins; drop each one once it is fixed):
//  - della-r3c* racks: MPI_Finalize (Open MPI 4.1.6 + UCX) intermittently
//    stalls ~52 s after any communication, running the allocation out of time.
//  - della-h17n1 (optane node): every multi-node job including it hangs in
//    MPI start-up until killed.
def excluded_nodes = 'della-r3c[1-4]n[1-16],della-h17n1'

pipeline {
    agent {
        node {
            label 'della-rse_specfempp'
        }
    }
    options {
        // Each pull request is its own multibranch sub-job, so this only aborts
        // the superseded build of the same pull request.
        disableConcurrentBuilds(abortPrevious: true)
    }
    stages {
        stage('Reset Reporting') {
            steps {
                // Every matrix cell shares one workspace and workspaces are reused between
                // builds, so a junit/ left by an earlier build would be archived as if it
                // belonged to this one.
                sh 'rm -rf junit && mkdir -p junit'
            }
        }
        stage('GCC OPENMPI Compiler Check') {
            matrix {
                axes {
                    axis {
                        name 'GNUCompiler'
                        values 'GCC11', 'GCC14'
                    }
                    axis {
                        name 'SIMD'
                        values 'SIMD_NONE', 'SIMD_NATIVE'
                    }
                    axis {
                        name 'HostSpace'
                        values 'SERIAL'
                    }
                    axis {
                        name 'MPIEnabled'
                        values 'MPI_GCC416'
                    }
                }
                // Read by .jenkins/cell.sh
                environment {
                    CELL            = "mpi_${GNUCompiler}_${MPIEnabled}_${HostSpace}_${SIMD}"
                    MODULES         = "${compiler_module[GNUCompiler]} ${mpi[MPIEnabled].module}"
                    CC              = 'mpicc'
                    CXX             = 'mpicxx'
                    FC              = 'mpifort'
                    CMAKE_FLAGS     = "${host_flags[HostSpace]} ${simd_flags[SIMD]} -DSPECFEM_ENABLE_MPI=ON -DSPECFEM_MPI_TEST_COMMAND=srun"
                    // salloc, not srun: ctest launches the ranks itself.
                    LAUNCHER        = 'salloc'
                    SLURM_FLAGS     = "${mpi[MPIEnabled].slurm} -t 00:15:00 --constraint=intel --exclude=${excluded_nodes}"
                    CTEST_FLAGS     = '-L MPI'
                    OMP_NUM_THREADS = '20'
                }
                stages {
                    stage('Build') {
                        steps {
                            sh '.jenkins/cell.sh build'
                        }
                    }
                    stage('Test') {
                        steps {
                            sh '.jenkins/cell.sh test'
                        }
                    }
                }
                post {
                    always {
                        sh '.jenkins/cell.sh clean'
                    }
                }
            }
        }
    }
    post {
        always {
            junit testResults: 'junit/*.xml',
                  allowEmptyResults: true,
                  skipPublishingChecks: true
            archiveArtifacts artifacts: 'junit/*.xml', allowEmptyArchive: true
        }
    }
}
