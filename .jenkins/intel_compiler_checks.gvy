// What each axis value means.

def compiler_module = [ICC2024: 'intel/2024.2']
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
        stage('Intel Host Compiler Check') {
            matrix {
                axes {
                    axis {
                        name 'IntelCompiler'
                        values 'ICC2024'
                    }
                    axis {
                        name 'HostSpace'
                        values 'SERIAL', 'OPENMP'
                    }
                    axis {
                        name 'SIMD'
                        values 'SIMD_NONE', 'SIMD_NATIVE'
                    }
                }
                // Read by .jenkins/cell.sh
                environment {
                    CELL            = "cpu_${IntelCompiler}_${HostSpace}_${SIMD}"
                    MODULES         = "${compiler_module[IntelCompiler]}"
                    CC              = 'icx'
                    CXX             = 'icpx'
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
