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
                        values 'SIMD_NONE;-DSPECFEM_ENABLE_SIMD=OFF', 'SIMD_NATIVE;-DSPECFEM_ENABLE_SIMD=ON -DKokkos_ARCH_SKX=ON -DKokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON'
                    }
                    axis{
                        name 'HostSpace'
                        values 'SERIAL;-DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_ATOMICS_BYPASS=ON;-n 1 -c 20;-j', 'OPENMP;-DKokkos_ENABLE_OPENMP=ON;-n 1 -c 20;-j 1'
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
                            CTEST_FLAGS = """${sh(
                                                    returnStdout: true,
                                                    script: 'cut -d";" -f4 <<<"${HostSpace}"'
                                                ).trim()}"""
                        }
                        stages {
                            stage (' Build '){
                                steps {
                                    echo "Building ${CMAKE_HOST_FLAGS} ${SIMD_FLAGS} with ${GNU_COMPILER_NAME}"
                                    sh """
                                        module load cmake/3.30.8
                                        module load boost/1.85.0
                                        module load ${GNU_COMPILER_MODULE}
                                        cmake3 -S . -B build_cpu_${GNU_COMPILER_NAME}_${CMAKE_HOST_NAME}_${SIMD_NAME}_${env.BUILD_TAG} \
                                          -DCMAKE_BUILD_TYPE=Release \
                                          -D CMAKE_INSTALL_PREFIX=install_cpu_${GNU_COMPILER_NAME}_${CMAKE_HOST_NAME}_${SIMD_NAME}_${env.BUILD_TAG}/bin \
                                          ${CMAKE_HOST_FLAGS} ${SIMD_FLAGS} \
                                          -D SPECFEM_BUILD_TESTS=ON \
                                          -D SPECFEMPP_TEST_DIR=/scratch/gpfs/TROMP/specfempp/jenkins/test_cpu_${GNU_COMPILER_NAME}_${CMAKE_HOST_NAME}_${SIMD_NAME}_${env.BUILD_TAG} \
                                          -D SPECFEM_BUILD_BENCHMARKS=OFF
                                        cmake3 --build build_cpu_${GNU_COMPILER_NAME}_${CMAKE_HOST_NAME}_${SIMD_NAME}_${env.BUILD_TAG}
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
                                        cd /scratch/gpfs/TROMP/specfempp/jenkins/test_cpu_${GNU_COMPILER_NAME}_${CMAKE_HOST_NAME}_${SIMD_NAME}_${env.BUILD_TAG} && \
                                        srun -N 1 -t 00:30:00 --account rse ${HOST_RUN_FLAGS} \
                                            --constraint="intel" \
                                            bash -c 'export OMP_PROC_BIND=spread; \
                                            export OMP_THREADS=places; ctest ${CTEST_FLAGS} --output-on-failure --no-tests=error;'
                                    """
                                    echo ' Testing completed '
                                }
                            }
                        }
                        post {
                            failure {
                                echo 'Build or Test stage failed, executing cleanup'
                                sh "rm -rf build_cpu_${GNU_COMPILER_NAME}_${CMAKE_HOST_NAME}_${SIMD_NAME}_${env.BUILD_TAG}"
                                sh "rm -rf install_cpu_${GNU_COMPILER_NAME}_${CMAKE_HOST_NAME}_${SIMD_NAME}_${env.BUILD_TAG}"
                                sh "rm -rf /scratch/gpfs/TROMP/specfempp/jenkins/test_cpu_${GNU_COMPILER_NAME}_${CMAKE_HOST_NAME}_${SIMD_NAME}_${env.BUILD_TAG}"
                            }
                        }
                    }
                }
            }
        }
    }
}
