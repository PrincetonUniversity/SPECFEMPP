// What each axis value means.
def compiler_module = [GCC8: 'gcc/11', GCC14: 'gcc-toolset/14']
def simd_flags = [
    SIMD_NONE  : '-DSPECFEM_ENABLE_SIMD=OFF',
    SIMD_NATIVE: '-DSPECFEM_ENABLE_SIMD=ON -DKokkos_ARCH_SKX=ON -DKokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON'
]
def host = [
    SERIAL: [cmake: '-DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_ATOMICS_BYPASS=ON', ctest: '-j'],
    OPENMP: [cmake: '-DKokkos_ENABLE_OPENMP=ON', ctest: '-j 1']
]

pipeline {
    agent {
        node {
            label 'della-rse_specfempp'
        }
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
        stage('GNU Host Compiler Check') {
            matrix {
                axes {
                    axis {
                        name 'GNUCompiler'
                        values 'GCC8', 'GCC14'
                    }
                    axis {
                        name 'SIMD'
                        values 'SIMD_NONE', 'SIMD_NATIVE'
                    }
                    axis {
                        name 'HostSpace'
                        values 'SERIAL', 'OPENMP'
                    }
                }
                // Read by .jenkins/cell.sh
                environment {
                    CELL            = "cpu_${GNUCompiler}_${HostSpace}_${SIMD}"
                    MODULES         = "${compiler_module[GNUCompiler]}"
                    CMAKE_FLAGS     = "${host[HostSpace].cmake} ${simd_flags[SIMD]}"
                    SLURM_FLAGS     = '-n 1 -c 4 -t 00:15:00 --constraint=intel'
                    CTEST_FLAGS     = "${host[HostSpace].ctest}"
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
