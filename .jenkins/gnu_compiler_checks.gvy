pipeline{
    agent {
        node {
            label 'della-rse_specfempp'
        }
    }
    stages{
        stage(' GNU Host Compiler Check '){
            matrix {
                axes {
                    axis{
                        name 'GNUCompiler'
                        values 'GCC8;gcc/11', 'GCC14;gcc-toolset/14'
                    }
                    axis{
                        name 'SIMD'
                        values 'SIMD_NONE;-DSPECFEM_ENABLE_SIMD=OFF', 'SIMD_NATIVE;-DSPECFEM_ENABLE_SIMD=ON -DKokkos_ARCH_NATIVE=ON -DKokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON'
                    }
                    axis{
                        name 'HostSpace'
                        values 'SERIAL;-DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_ATOMICS_BYPASS=ON;-n 1 -c 20', 'OPENMP;-DKokkos_ENABLE_OPENMP=ON;-n 1 -c 20'
                    }
                    axis{
                        name 'MPIEnabled'
                        values 'MPI_OFF;-DSPECFEM_ENABLE_MPI=OFF;', 'OPENMPI_4.1.6;-DSPECFEM_ENABLE_MPI=ON;openmpi/gcc/4.1.6'
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
                            MPI_NAME = """${sh(
                                                    returnStdout: true,
                                                    script: 'cut -d";" -f1 <<<"${MPIEnabled}"'
                                                ).trim()}"""
                            MPI_BUILD = """${sh(
                                                    returnStdout: true,
                                                    script: 'cut -d";" -f2 <<<"${MPIEnabled}"'
                                                ).trim()}"""
                            MPI_MODULE = """${sh(
                                                    returnStdout: true,
                                                    script: 'cut -d";" -f3 <<<"${MPIEnabled}"'
                                                ).trim()}"""
                        }
                        stages {
                            stage (' Build '){
                                steps {
                                    echo "Building ${CMAKE_HOST_FLAGS} ${SIMD_FLAGS} with ${GNU_COMPILER_NAME} and ${MPI_NAME}"
                                    sh """
                                        module load cmake/3.30.8
                                        module load boost/1.85.0
                                        module load ${GNU_COMPILER_MODULE}
                                        module load ${MPI_MODULE}
                                        cmake3 -S . -B build_cpu_${GNU_COMPILER_NAME}_${CMAKE_HOST_NAME}_${SIMD_NAME}_${MPI_NAME}_${env.BUILD_TAG} \
                                          -DCMAKE_BUILD_TYPE=Release \
                                          -D CMAKE_INSTALL_PREFIX=install_cpu_${GNU_COMPILER_NAME}_${CMAKE_HOST_NAME}_${SIMD_NAME}_${MPI_NAME}_${env.BUILD_TAG}/bin \
                                          ${CMAKE_HOST_FLAGS} ${SIMD_FLAGS} ${MPI_BUILD} \
                                          -D SPECFEM_BUILD_TESTS=ON \
                                          -D SPECFEM_BUILD_BENCHMARKS=OFF
                                        cmake3 --build build_cpu_${GNU_COMPILER_NAME}_${CMAKE_HOST_NAME}_${SIMD_NAME}_${MPI_NAME}_${env.BUILD_TAG}
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
                                        module load ${GNU_COMPILER_MODULE}
                                        module load ${MPI_MODULE}
                                        cd build_cpu_${GNU_COMPILER_NAME}_${CMAKE_HOST_NAME}_${SIMD_NAME}_${MPI_NAME}_${env.BUILD_TAG}/tests/unit-tests
                                        export BUILD_DIR=build_cpu_${GNU_COMPILER_NAME}_${CMAKE_HOST_NAME}_${SIMD_NAME}_${MPI_NAME}_${env.BUILD_TAG}
                                        srun -N 1 -t 00:30:00 --account rse ${HOST_RUN_FLAGS} \
                                            --constraint="intel|cascade" \
                                            bash -c 'export OMP_PROC_BIND=spread; \
                                            export OMP_THREADS=places; ctest -j --output-on-failure;'
                                    """
                                    echo ' Testing completed '
                                }
                            }
                        }
                        post {
                            always {
                                echo ' Cleaning '
                                sh "rm -rf build_cpu_${GNU_COMPILER_NAME}_${CMAKE_HOST_NAME}_${SIMD_NAME}_${MPI_NAME}_${env.BUILD_TAG}"
                                sh "rm -rf install_cpu_${GNU_COMPILER_NAME}_${CMAKE_HOST_NAME}_${SIMD_NAME}_${MPI_NAME}_${env.BUILD_TAG}"
                            }
                        }
                    }
                }
            }
        }
    }
}
