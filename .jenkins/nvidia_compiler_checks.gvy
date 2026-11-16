// What each axis value means.

def cuda_module = [CUDA126: 'cudatoolkit/12.6', CUDA129: 'cudatoolkit/12.9']
// SERIAL runs tests in parallel; OPENMP uses -j 1 to avoid oversubscription.
def host = [
    SERIAL: [cmake: '-DKokkos_ENABLE_SERIAL=ON', ctest: '-j 4'],
    OPENMP: [cmake: '-DKokkos_ENABLE_OPENMP=ON', ctest: '-j 1']
]
def device = [
    CUDA_AMPERE80: [cmake: '-DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON',
                    slurm: '--gres=gpu:1 --constraint=a100']
]
def simd_flags = [
    SIMD_NONE  : '-DSPECFEM_ENABLE_SIMD=OFF',
    SIMD_NATIVE: '-DSPECFEM_ENABLE_SIMD=ON -DKokkos_ARCH_AMDAVX=ON'
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
        stage('NVIDIA Device Compiler Check') {
            matrix {
                axes {
                    axis {
                        name 'CUDACompiler'
                        values 'CUDA126', 'CUDA129'
                    }
                    axis {
                        name 'HostSpace'
                        values 'SERIAL', 'OPENMP'
                    }
                    axis {
                        name 'DeviceSpace'
                        values 'CUDA_AMPERE80'
                    }
                    axis {
                        name 'SIMD'
                        values 'SIMD_NONE', 'SIMD_NATIVE'
                    }
                }
                // Read by .jenkins/cell.sh
                environment {
                    CELL            = "cuda_${CUDACompiler}_${HostSpace}_${DeviceSpace}_${SIMD}"
                    MODULES         = "${cuda_module[CUDACompiler]}"
                    CMAKE_FLAGS     = "${host[HostSpace].cmake} ${device[DeviceSpace].cmake} ${simd_flags[SIMD]}"
                    SLURM_FLAGS     = "-N 1 -n 1 -c 20 -t 00:35:00 ${device[DeviceSpace].slurm}"
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
