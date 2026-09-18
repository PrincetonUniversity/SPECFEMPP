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
                        values 'SIMD_NONE;-DSPECFEM_ENABLE_SIMD=OFF', 'SIMD_NATIVE;-DSPECFEM_ENABLE_SIMD=ON -DKokkos_ARCH_SKX=ON -DKokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON'
                    }
                    axis{
                        name 'HostSpace'
                        values 'SERIAL;-DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_ATOMICS_BYPASS=ON'
                    }
                    axis{
                        name 'MPIEnabled'
                        values 'MPI_GCC416;-DSPECFEM_ENABLE_MPI=ON;openmpi/gcc/4.1.6;--ntasks=8  --nodes=2-8'
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
                                          -D SPECFEM_BUILD_BENCHMARKS=OFF \
                                          -D SPECFEM_MPI_TEST_COMMAND=srun \
                                          -D SPECFEMPP_TEST_DIR=/scratch/gpfs/TROMP/specfempp/jenkins/test_mpi_${GNU_COMPILER_NAME}_${MPI_NAME}_${CMAKE_HOST_NAME}_${SIMD_NAME}_${env.BUILD_TAG}
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
                                        cd /scratch/gpfs/TROMP/specfempp/jenkins/test_mpi_${GNU_COMPILER_NAME}_${MPI_NAME}_${CMAKE_HOST_NAME}_${SIMD_NAME}_${env.BUILD_TAG}
                                        salloc ${MPI_RUN_FLAGS} -t 00:15:00 --account rse \
                                            --constraint="intel" \
                                            bash -c 'export OMP_PROC_BIND=spread; \
                                            export OMP_PLACES=threads; \
                                            export OMP_NUM_THREADS=20; \
                                            hostname; \
                                            echo "ranks run on: \$SLURM_JOB_NODELIST"; \
                                            ctest -L MPI --output-on-failure --no-tests=error;'
                                    """
                                    echo ' Testing completed '
                                }
                            }
                        }
                        post {
                            always {
                                echo 'Executing cleanup of build and test artifacts'
                                sh "rm -rf build_mpi_${GNU_COMPILER_NAME}_${MPI_NAME}_${CMAKE_HOST_NAME}_${SIMD_NAME}_${env.BUILD_TAG}"
                                sh "rm -rf install_mpi_${GNU_COMPILER_NAME}_${MPI_NAME}_${CMAKE_HOST_NAME}_${SIMD_NAME}_${env.BUILD_TAG}"
                                sh "rm -rf /scratch/gpfs/TROMP/specfempp/jenkins/test_mpi_${GNU_COMPILER_NAME}_${MPI_NAME}_${CMAKE_HOST_NAME}_${SIMD_NAME}_${env.BUILD_TAG}"
                            }
                        }
                    }
                }
            }
        }
    }
}
