pipeline{
    agent {
        node {
            label 'della_rk9481'
        }
    }
    stages{
        stage(' Intel Host Compiler Check '){
            matrix {
                axes {
                    axis{
                        name 'IntelCompiler'
                        values 'ICC2024;intel/2024.2'
                    }
                    axis{
                        name 'HostSpace'
                        values 'SERIAL;-DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_ATOMICS_BYPASS=ON;-n 1', 'OPENMP;-DKokkos_ENABLE_OPENMP=ON;-n 1 -c 10'
                    }
                    axis{
                        name 'SIMD'
                        values 'SIMD_NONE;-DENABLE_SIMD=OFF', 'SIMD_NATIVE;-DENABLE_SIMD=ON -DKokkos_ARCH_NATIVE=ON -DKokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON'
                    }
                }
                stages {
                    stage ('Build and Clean '){
                        environment {
                            INTEL_COMPILER_NAME = """${sh(
                                                    returnStdout: true,
                                                    script: 'cut -d";" -f1 <<<"${IntelCompiler}"'
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
                            HOST_RUN_FLAGS = """${sh(
                                                    returnStdout: true,
                                                    script: 'cut -d";" -f3 <<<"${HostSpace}"'
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
                                    echo "Building ${CMAKE_HOST_FLAGS} ${SIMD_FLAGS} with ${INTEL_COMPILER_NAME}"
                                    sh """
                                        module load boost/1.85.0
                                        module load ${INTEL_MODULE}
                                        export CC=icx
                                        export CXX=icpx
                                        cmake3 -S . -B build_cpu_${INTEL_COMPILER_NAME}_${CMAKE_HOST_NAME}_${SIMD_NAME}_${env.BUILD_TAG} -DCMAKE_BUILD_TYPE=Release ${CMAKE_HOST_FLAGS} ${SIMD_FLAGS} -D BUILD_TESTS=ON -D BUILD_EXAMPLES=OFF
                                        cmake3 --build build_cpu_${INTEL_COMPILER_NAME}_${CMAKE_HOST_NAME}_${SIMD_NAME}_${env.BUILD_TAG}
                                    """
                                    echo ' Build completed '
                                }
                            }
                            stage (' Test '){
                                steps {
                                    echo ' Testing '
                                    sh """
                                        module load boost/1.85.0
                                        module load ${INTEL_MODULE}
                                        cd build_cpu_${INTEL_COMPILER_NAME}_${CMAKE_HOST_NAME}_${SIMD_NAME}_${env.BUILD_TAG}/tests/unit-tests
                                        srun -N 1 -t 00:20:00 ${HOST_RUN_FLAGS} --constraint=skylake bash -c 'export OMP_PROC_BIND=spread; export OMP_THREADS=places; ctest --verbose;'
                                    """
                                    echo ' Testing completed '
                                }
                            }
                        }
                        post {
                            always {
                                echo ' Cleaning '
                                sh "rm -rf build_cpu_${INTEL_COMPILER_NAME}_${CMAKE_HOST_NAME}_${SIMD_NAME}_${env.BUILD_TAG}"
                            }
                        }
                    }
                }
            }
        }
    }
}
