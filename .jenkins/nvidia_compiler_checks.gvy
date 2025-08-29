pipeline{
    agent {
        node {
            label 'della-rse_specfempp'
        }
    }
    stages{
        stage(' NVIDIA Device Compiler Check '){
            matrix {
                axes {
                    axis{
                        name 'CUDACompiler'
                        values 'CUDA117;cudatoolkit/11.8', 'CUDA126;cudatoolkit/12.8'
                    }
                    axis{
                        name 'HostSpace'
                        values 'SERIAL;-DKokkos_ENABLE_SERIAL=ON;-n 1 -c 20', 'OPENMP;-DKokkos_ENABLE_OPENMP=ON; -n 1 -c 20'
                    }
                    axis{
                        name 'DeviceSpace'
                        values 'CUDA_AMPERE80;-DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON; --gres=gpu:1 --constraint=a100'
                    }
                    axis{
                        name 'SIMD'
                        values 'SIMD_NONE;-DSPECFEM_ENABLE_SIMD=OFF', 'SIMD_NATIVE;-DSPECFEM_ENABLE_SIMD=ON -DKokkos_ARCH_AMDAVX=ON'
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
                                        module load cmake/3.30.8
                                        module load boost/1.85.0
                                        module load ${CUDA_MODULE}
                                        cmake3 -S . -B build_cuda_${CUDA_COMPILER_NAME}_${CMAKE_HOST_NAME}_${CMAKE_DEVICE_NAME}_${SIMD_NAME}_${env.BUILD_TAG} -DCMAKE_BUILD_TYPE=Release ${CMAKE_HOST_FLAGS} ${CMAKE_DEVICE_FLAGS} ${SIMD_FLAGS} -D SPECFEM_BUILD_TESTS=ON -D SPECFEM_BUILD_BENCHMARKS=OFF
                                        cmake3 --build build_cuda_${CUDA_COMPILER_NAME}_${CMAKE_HOST_NAME}_${CMAKE_DEVICE_NAME}_${SIMD_NAME}_${env.BUILD_TAG}
                                    """
                                    echo ' Build completed '
                                }
                            }
                            stage (' Test '){
                                steps {
                                    echo ' Testing '
                                    sh """
                                        module load cmake/3.30.8
                                        module load boost/1.85.0
                                        module load ${CUDA_MODULE}
                                        cd build_cuda_${CUDA_COMPILER_NAME}_${CMAKE_HOST_NAME}_${CMAKE_DEVICE_NAME}_${SIMD_NAME}_${env.BUILD_TAG}/tests/unit-tests
                                        export BUILD_DIR=build_cuda_${CUDA_COMPILER_NAME}_${CMAKE_HOST_NAME}_${CMAKE_DEVICE_NAME}_${SIMD_NAME}_${env.BUILD_TAG}
                                        srun -N 1 -t 00:30:00 --account rse ${HOST_RUN_FLAGS} ${DEVICE_RUN_FLAGS} bash -c 'export OMP_PROC_BIND=spread; export OMP_THREADS=places; ctest -j 10 --output-on-failure;'
                                    """
                                    echo ' Testing completed '
                                }
                            }
                        }
                        post {
                            always {
                                echo ' Cleaning '
                                sh "rm -rf build_cuda_${CUDA_COMPILER_NAME}_${CMAKE_HOST_NAME}_${CMAKE_DEVICE_NAME}_${SIMD_NAME}_${env.BUILD_TAG}"
                            }
                        }
                    }
                }
            }

        }
    }
}
