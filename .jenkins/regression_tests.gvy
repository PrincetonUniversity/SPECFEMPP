pipeline{
    agent {
        node {
            label 'della_rk9481'
        }
    }

    stages {
        stage (' Build main branch '){
            when {
                branch env.BRANCH_NAME
            }
            matrix {
                axes {
                    axis{
                        name 'HostSpace'
                        values 'SERIAL;-DKokkos_ENABLE_SERIAL=ON;-n 1', 'OPENMP;-DKokkos_ENABLE_OPENMP=ON;-n 10'
                    }
                    axis{
                        name 'DeviceSpace'
                        values 'NONE;-DKokkos_ENABLE_CUDA=OFF;--constraint=skylake', 'CUDA_AMPERE80;-DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON;--gres=gpu:1 --constraint=a100'
                    }
                }
                stages {
                    stage( ' Load git modules ' ){
                        steps {
                            echo ' Getting git submodules '
                            sh 'git submodule init'
                            sh 'git submodule update'
                        }
                    }
                    stage ( ' Load required modules ' ){
                        steps {
                            echo ' Loading required module '
                            sh 'module load cudatoolkit/11.7'
                        }
                    }
                    stage ( ' Build ' ){
                        environment {
                            // Define C compiler
                            CC = 'cc'
                            // Using CXX compiler
                            CXX = 'c++'
                            // Define cuda root directory
                            CUDA_ROOT='/usr/local/cuda-11.7'
                            // CMAKE build flags
                            CMAKE_HOST_NAME = """${sh(
                                                returnStdout: true,
                                                script: 'cut -d";" -f1 <<<"${HostSpace}"'
                                            ).trim()}"""
                            CMAKE_HOST_FLAGS = """${sh(
                                                returnStdout: true,
                                                script: 'cut -d";" -f2 <<<"${HostSpace}"'
                                            ).trim()}"""
                            CMAKE_DEVICE_NAME = """${sh(
                                                    returnStdout: true,
                                                    script: 'cut -d";" -f1 <<<"${DeviceSpace}"'
                                                ).trim()}"""
                            CMAKE_DEVICE_FLAGS = """${sh(
                                                    returnStdout: true,
                                                    script: 'cut -d";" -f2 <<<"${DeviceSpace}"'
                                                ).trim()}"""
                        }
                        steps {
                            echo "Building ${CMAKE_HOST_FLAGS} ${CMAKE_DEVICE_FLAGS}"
                            sh "cmake3 -S . -B build_GNU_${CMAKE_HOST_NAME}_${CMAKE_DEVICE_NAME} -DCMAKE_BUILD_TYPE=Release ${CMAKE_HOST_FLAGS} ${CMAKE_DEVICE_FLAGS}"
                            sh 'cmake3 --build build'
                            echo ' Build completed '
                        }
                    }
                    stage ( ' Run Regression Tests '){
                        environment {
                            CMAKE_HOST_NAME = """${sh(
                                                returnStdout: true,
                                                script: 'cut -d";" -f1 <<<"${HostSpace}"'
                                            ).trim()}"""
                            CMAKE_DEVICE_NAME = """${sh(
                                                    returnStdout: true,
                                                    script: 'cut -d";" -f1 <<<"${DeviceSpace}"'
                                                ).trim()}"""
                            SLURM_RUN_HOST_CONFIG = """${sh(
                                                        returnStdout: true,
                                                        script: 'cut -d";" -f3 <<<"${HostSpace}"'
                                                    ).trim()}"""
                            SLURM_RUN_DEVICE_CONFIG = """${sh(
                                                        returnStdout: true,
                                                        script: 'cut -d";" -f3 <<<"${DeviceSpace}"'
                                                    ).trim()}"""
                        }
                        steps {
                            echo "Host Name = ${CMAKE_HOST_NAME}, Device Name = ${CMAKE_DEVICE_NAME}, Host Config = ${SLURM_RUN_HOST_CONFIG}, Device Config = ${SLURM_RUN_DEVICE_CONFIG}"
                            // sh "srun -N 1 -t 01:00:00 ${SLURM_RUN_HOST_CONFIG} ${SLURM_RUN_DEVICE_CONFIG} -J Jenkins_main_RT_${CMAKE_HOST_NAME}_${CMAKE_DEVICE_NAME} ./build_GNU_${CMAKE_HOST_NAME}_${CMAKE_DEVICE_NAME}/specfem2d -p regression_tests/elastic_domain/specfem_config.yaml"
                        }
                    }
                    stage ( ' Clean ' ){
                        environment {
                            // CMAKE build flags
                            CMAKE_HOST_NAME = """${sh(
                                                    returnStdout: true,
                                                    script: 'cut -d";" -f1 <<<"${HostSpace}"'
                                                ).trim()}"""
                            CMAKE_DEVICE_NAME = """${sh(
                                                    returnStdout: true,
                                                    script: 'cut -d";" -f1 <<<"${DeviceSpace}"'
                                                ).trim()}"""
                        }
                        steps {
                            echo ' Cleaning '
                            sh "rm -rf build_GNU_${CMAKE_HOST_NAME}_${CMAKE_DEVICE_NAME}"
                        }
                    }
                }
            }
        }
    }
}
