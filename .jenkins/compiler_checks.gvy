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
        parallel {
            stage(' GNU Host Compiler Check '){
                matrix {
                    axes {
                        axis{
                            name 'GNUCompiler'
                            values 'GCC8;gcc/8', 'GCC13;gcc-toolset/13'
                        }
                        axis{
                            name 'SIMD'
                            values 'SIMD_NONE;-DENABLE_SIMD=OFF', 'SIMD_NATIVE;-DENABLE_SIMD=ON -DKokkos_ARCH_NATIVE=ON -DKokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON'
                        }
                        axis{
                            name 'HostSpace'
                            values 'SERIAL;-DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_ATOMICS_BYPASS=ON', 'OPENMP;-DKokkos_ENABLE_OPENMP=ON'
                        }
                    }
                    stages {
                        stage ('Build and Clean '){
                            environment {
                                // CMAKE build flags
                                GNU_COMPILER_NAME = """${sh(
                                                        returnStdout: true,
                                                        script: 'cut -d";" -f1 <<<"${GNUCompiler}"'
                                                    ).trim()}"""
                                GNU_COMPILER_MODULE = """${sh(
                                                        returnStdout: true,
                                                        script: 'cut -d";" -f2 <<<"${GNUCompiler}"'
                                                    ).trim()}"""
                                CMAKE_HOST_NAME = """${sh(
                                                        returnStdout: true,
                                                        script: 'cut -d";" -f1 <<<"${HostSpace}"'
                                                    ).trim()}"""
                                CMAKE_HOST_FLAGS = """${sh(
                                                        returnStdout: true,
                                                        script: 'cut -d";" -f2 <<<"${HostSpace}"'
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
                                        echo "Building ${CMAKE_HOST_FLAGS} ${CMAKE_DEVICE_FLAGS} ${SIMD_FLAGS} with ${GNU_COMPILER_NAME}"
                                        sh """
                                            module load boost/1.73.0
                                            module load ${GNU_COMPILER_MODULE}
                                            cmake3 -S . -B build_cpu_{GNU_COMPILER_NAME}_${CMAKE_HOST_NAME}_${SIMD_NAME}_${env.GIT_COMMIT} -DCMAKE_BUILD_TYPE=Release ${CMAKE_HOST_FLAGS} ${SIMD_FLAGS}
                                            cmake3 --build build_cpu_{GNU_COMPILER_NAME}_${CMAKE_HOST_NAME}_${SIMD_NAME}_${env.GIT_COMMIT}
                                        """
                                        echo ' Build completed '
                                    }
                                }
                            }
                            post {
                                always {
                                    echo ' Cleaning '
                                    sh "rm -rf build_cpu_{GNU_COMPILER_NAME}_${CMAKE_HOST_NAME}_${SIMD_NAME}_${env.GIT_COMMIT}"
                                }
                            }
                        }
                    }
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
                            values 'SERIAL;-DKokkos_ENABLE_SERIAL=ON', 'OPENMP;-DKokkos_ENABLE_OPENMP=ON'
                        }
                        axis{
                            name 'DeviceSpace'
                            values 'CUDA_AMPERE80;-DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON', 'CUDA_VOLTA70;-DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_VOLTA70=ON'
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
                                                        script: 'cut -d";" -
                                                        f1 <<<"${HostSpace}"'
                                                    ).trim()}"""
                                CMAKE_HOST_FLAGS = """${sh(
                                                        returnStdout: true,
                                                        script: 'cut -d";" -
                                                        f2 <<<"${HostSpace}"'
                                                    ).trim()}"""
                                CMAKE_DEVICE_NAME = """${sh(
                                                        returnStdout: true,
                                                        script: 'cut -d";" -
                                                        f1 <<<"${DeviceSpace}"'
                                                    ).trim()}"""
                                CMAKE_DEVICE_FLAGS = """${sh(
                                                        returnStdout: true,
                                                        script: 'cut -d";" -
                                                        f2 <<<"${DeviceSpace}"'
                                                    ).trim()}"""
                                SIMD_NAME = """${sh(
                                                        returnStdout: true,
                                                        script: 'cut -d";" -
                                                        f1 <<<"${SIMD}"'
                                                    ).trim()}"""
                                SIMD_FLAGS = """${sh(
                                                        returnStdout: true,
                                                        script: 'cut -d";" -
                                                        f2 <<<"${SIMD}"'
                                                    ).trim()}"""
                            }
                            stages {
                                stage (' Build '){
                                    steps {
                                        echo "Building ${CMAKE_HOST_FLAGS} ${CMAKE_DEVICE_FLAGS} ${SIMD_FLAGS} with ${CUDA_COMPILER_NAME}"
                                        sh """
                                            module load boost/1.73.0
                                            module load ${CUDA_MODULE}
                                            cmake3 -S . -B build_cuda_${CUDA_COMPILER_NAME}_${CMAKE_HOST_NAME}_${CMAKE_DEVICE_NAME}_${SIMD_NAME}_${env.GIT_COMMIT} -DCMAKE_BUILD_TYPE=Release ${CMAKE_HOST_FLAGS} ${CMAKE_DEVICE_FLAGS} ${SIMD_FLAGS}
                                            cmake3 --build build_cuda_${CUDA_COMPILER_NAME}_${CMAKE_HOST_NAME}_${CMAKE_DEVICE_NAME}_${SIMD_NAME}_${env.GIT_COMMIT}
                                        """
                                        echo ' Build completed '
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
            stage(' Intel Host Compiler Check '){
                matrix {
                    axes {
                        axis{
                            name 'IntelCompiler'
                            values 'ICC2022;intel/2022.2.0'
                        }
                        axis{
                            name 'HostSpace'
                            values 'SERIAL;-DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_ATOMICS_BYPASS=ON', 'OPENMP;-DKokkos_ENABLE_OPENMP=ON'
                        }
                        axis{
                            name 'SIMD'
                            values 'SIMD_NONE;-DENABLE_SIMD=OFF', 'SIMD_NATIVE;-DENABLE_SIMD=ON -DKokkos_ARCH_NATIVE=ON'
                        }
                    }
                    stages {
                        stage ('Build and Clean '){
                            environment {
                                INTEL_COMPILER_NAME = """${sh(
                                                        returnStdout: true,
                                                        script: 'cut -d";" <<<"${IntelCompiler}"'
                                                    ).trim()}"""
                                INTEL_MODULE = """${sh(
                                                        returnStdout: true,
                                                        script: 'cut -d";" -f2 <<<"${IntelCompiler}"'
                                                    ).trim()}"""
                                CMAKE_HOST_NAME = """${sh(
                                                        returnStdout: true,
                                                        script: 'cut -d";" -f1 <<<"${HostSpace}"'
                                                    ).trim()}"""
                                CMAKE_HOST_FLAGS = """${sh(
                                                        returnStdout: true,
                                                        script: 'cut -d";" -f2 <<<"${HostSpace}"'
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
                                        echo "Building ${CMAKE_HOST_FLAGS} ${CMAKE_DEVICE_FLAGS} ${SIMD_FLAGS} with ${INTEL_COMPILER_NAME}"
                                        sh """
                                            module load boost/1.73.0
                                            module load ${INTEL_MODULE}
                                            export CC=icx
                                            export CXX=icpx
                                            cmake3 -S . -B build_cpu_${INTEL_COMPILER_NAME}_${CMAKE_HOST_NAME}_${SIMD_NAME}_${env.GIT_COMMIT} -DCMAKE_BUILD_TYPE=Release ${CMAKE_HOST_FLAGS} ${SIMD_FLAGS}
                                            cmake3 --build build_cpu_${INTEL_COMPILER_NAME}_${CMAKE_HOST_NAME}_${SIMD_NAME}_${env.GIT_COMMIT}
                                        """
                                        echo ' Build completed '
                                    }
                                }
                            }
                            post {
                                always {
                                    echo ' Cleaning '
                                    sh "rm -rf build_cpu_${INTEL_COMPILER_NAME}_${CMAKE_HOST_NAME}_${SIMD_NAME}_${env.GIT_COMMIT}"
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
