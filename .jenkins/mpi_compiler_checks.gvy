pipeline {
    agent {
        node {
            label 'della-rse_specfempp'
        }
    }
    stages{
        stage(' GCC OPENMPI Compiler Check '){
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
                        values 'MPI_GCC416;-DSPECFEM_ENABLE_MPI=ON;openmpi/gcc/4.1.6;-N 2 -c 20'
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
                            MPI_FLAGS = """${sh(
                                                    returnStdout: true,
                                                    script: 'cut -d";" -f2 <<<"${MPIEnabled}"'
                                                ).trim()}"""
                            MPI_MODULE = """${sh(
                                                    returnStdout: true,
                                                    script: 'cut -d";" -f3 <<<"${MPIEnabled}"'
                                                ).trim()}"""
                            MPI_RUN_FLAGS = """${sh(
                                                    returnStdout: true,
                                                    script: 'cut -d";" -f4 <<<"${MPIEnabled}"'
                                                ).trim()}"""
                        }
                        stages {
                            stage (' Build '){
                                steps {
                                    echo "Building ${CMAKE_HOST_FLAGS} ${SIMD_FLAGS} ${MPI_FLAGS} with ${GNU_COMPILER_NAME} and ${MPI_NAME}"
                                    sh """
                                        module load cmake/3.30.8
                                        module load boost/1.85.0
                                        module load ${GNU_COMPILER_MODULE}
                                        module load ${MPI_MODULE}
                                        export CC=mpicc
                                        export CXX=mpicxx
                                        export FC=mpifort
                                        cmake3 -S . -B build_mpi_${GNU_COMPILER_NAME}_${MPI_NAME}_${CMAKE_HOST_NAME}_${SIMD_NAME}_${env.BUILD_TAG} \
                                          -DCMAKE_BUILD_TYPE=Release \
                                          -D CMAKE_INSTALL_PREFIX=install_mpi_${GNU_COMPILER_NAME}_${MPI_NAME}_${CMAKE_HOST_NAME}_${SIMD_NAME}_${env.BUILD_TAG}/bin \
                                          ${CMAKE_HOST_FLAGS} ${SIMD_FLAGS} ${MPI_FLAGS} \
                                          -D SPECFEM_BUILD_TESTS=ON \
                                          -D SPECFEM_BUILD_BENCHMARKS=OFF
                                        cmake3 --build build_mpi_${GNU_COMPILER_NAME}_${MPI_NAME}_${CMAKE_HOST_NAME}_${SIMD_NAME}_${env.BUILD_TAG}
                                    """
                                    echo ' Build completed '
                                }
                            }
                            stage (' Test '){
                                steps {
                                    echo ' Running MPI tests with ctest '
                                    sh """
                                        module load cmake/3.30.8
                                        module load boost/1.85.0
                                        module load ${GNU_COMPILER_MODULE}
                                        module load ${MPI_MODULE}
                                        cd build_mpi_${GNU_COMPILER_NAME}_${MPI_NAME}_${CMAKE_HOST_NAME}_${SIMD_NAME}_${env.BUILD_TAG}/tests/unit-tests
                                        export BUILD_DIR=build_mpi_${GNU_COMPILER_NAME}_${MPI_NAME}_${CMAKE_HOST_NAME}_${SIMD_NAME}_${env.BUILD_TAG}
                                        srun -N 1 --ntasks 4 -t 00:30:00 --account rse ctest --output-on-failure -j 4
                                    """
                                    echo ' Testing completed '
                                }
                            }
                        }
                        post {
                            always {
                                echo ' Cleaning '
                                sh "rm -rf build_mpi_${GNU_COMPILER_NAME}_${MPI_NAME}_${CMAKE_HOST_NAME}_${SIMD_NAME}_${env.BUILD_TAG}"
                                sh "rm -rf install_mpi_${GNU_COMPILER_NAME}_${MPI_NAME}_${CMAKE_HOST_NAME}_${SIMD_NAME}_${env.BUILD_TAG}"
                            }
                        }
                    }
                }
            }
        }
    }
}
