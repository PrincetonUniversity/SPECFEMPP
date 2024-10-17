pipeline{
    agent {
        node {
            label 'della_rk9481'
        }
    }
    stages{
        stage( ' Load git modules ' ){
            steps {
                echo ' Getting git submodules '
                sh 'git submodule init'
                sh 'git submodule update'
            }
        }
        stage(' NVIDIA Device Compiler Check '){
            matrix {
                axes {
                    axis{
                        name 'CUDACompiler'
                        values 'CUDA117;cudatoolkit/11.7', 'CUDA126;cudatoolkit/12.6'
                    }
                    axis{
                        name 'HostSpace'
                        values 'SERIAL;-DKokkos_ENABLE_SERIAL=ON;-n 1', 'OPENMP;-DKokkos_ENABLE_OPENMP=ON;-n 10'
                    }
                    axis{
                        name 'DeviceSpace'
                        values 'CUDA_AMPERE80;-DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON;--gres=gpu:1 --constraint=a100', 'CUDA_VOLTA70;-DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_VOLTA70=ON;--gres=gpu:1 --constraint=v100'
                    }
                    axis{
                        name 'SIMD'
                        values 'SIMD_NONE;-DENABLE_SIMD=OFF', 'SIMD_NATIVE;-DENABLE_SIMD=ON -DKokkos_ARCH_NATIVE=ON'
                    }
                }
                stages {
                    stage ('Build and Clean '){
                        environment {
                            CUDA_COMPILER_NAME = """${sh(
                                                    returnStdout: true,
                                                    script: 'cut -d";" -f1 <<<"${CUDACompiler}"'
                                                ).trim()}"""
                            CUDA_MODULE = """${sh(
                                                    returnStdout: true,
                                                    script: 'cut -d";" -f2 <<<"${CUDACompiler}"'
                                                ).trim()}"""
                            CMAKE_HOST_NAME = """${sh(
                                                    returnStdout: true,
                                                    script: 'cut -d";" -f1 <<<"${HostSpace}"'
                                                ).trim()}"""
                            CMAKE_HOST_FLAGS = """${sh(
                                                    returnStdout: true,
                                                    script: 'cut -d";" -f2 <<<"${HostSpace}"'
                                                ).trim()}"""
                            HOST_RUN_FLAGS = """${sh(
                                                    returnStdout: true,
                                                    script: 'cut -d";" -f3 <<<"${HostSpace}"'
                                                ).trim()}"""
                            CMAKE_DEVICE_NAME = """${sh(
                                                    returnStdout: true,
                                                    script: 'cut -d";" -f1 <<<"${DeviceSpace}"'
                                                ).trim()}"""
                            CMAKE_DEVICE_FLAGS = """${sh(
                                                    returnStdout: true,
                                                    script: 'cut -d";" -f2 <<<"${DeviceSpace}"'
                                                ).trim()}"""
                            DEVICE_RUN_FLAGS = """${sh(
                                                    returnStdout: true,
                                                    script: 'cut -d";" -f3 <<<"${DeviceSpace}"'
                                                ).trim()}"""
                            SIMD_NAME = """${sh(
                                                    returnStdout: true,
                                                    script: 'cut -d";" -f1 <<<"${SIMD}"'
                                                ).trim()}"""
                            SIMD_FLAGS = """${sh(
                                                    returnStdout: true,
                                                    script: 'cut -d";" -f2 <<<"${SIMD}"'
                                                ).trim()}"""
                        }
                        stages {
                            stage (' Build '){
                                steps {
                                    echo "Building ${CMAKE_HOST_FLAGS} ${CMAKE_DEVICE_FLAGS} ${SIMD_FLAGS} with ${CUDA_COMPILER_NAME}"
                                    sh """
                                        module load boost/1.73.0
                                        module load ${CUDA_MODULE}
                                        cmake3 -S . -B build_cuda_${CUDA_COMPILER_NAME}_${CMAKE_HOST_NAME}_${CMAKE_DEVICE_NAME}_${SIMD_NAME}_${env.GIT_COMMIT} -DCMAKE_BUILD_TYPE=Release ${CMAKE_HOST_FLAGS} ${CMAKE_DEVICE_FLAGS} ${SIMD_FLAGS} -D BUILD_TESTS=ON
                                        cmake3 --build build_cuda_${CUDA_COMPILER_NAME}_${CMAKE_HOST_NAME}_${CMAKE_DEVICE_NAME}_${SIMD_NAME}_${env.GIT_COMMIT}
                                    """
                                    echo ' Build completed '
                                }
                            }
                            stage (' Test '){
                                steps {
                                    echo ' Testing '
                                    sh """
                                        module load boost/1.73.0
                                        module load ${CUDA_MODULE}
                                        cd build_cpu_${CUDA_COMPILER_NAME}_${CMAKE_HOST_NAME}_${SIMD_NAME}_${env.GIT_COMMIT}/tests/unit-tests
                                        srun -N 1 -t 00:20:00 ${HOST_RUN_FLAGS} ${DEVICE_RUN_FLAGS} bash -c 'export OMP_PROC_BIND=spread; export OMP_THREADS=places; ctest --verbose;'
                                    """
                                    echo ' Testing completed '
                                }
                            }
                        }
                        post {
                            always {
                                echo ' Cleaning '
                                sh "rm -rf build_cuda_${CUDA_COMPILER_NAME}_${CMAKE_HOST_NAME}_${CMAKE_DEVICE_NAME}_${SIMD_NAME}_${env.GIT_COMMIT}"
                            }
                        }
                    }
                }
            }

        }
    }
}
