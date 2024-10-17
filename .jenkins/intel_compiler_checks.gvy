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
        stage(' Intel Host Compiler Check '){
            matrix {
                axes {
                    axis{
                        name 'IntelCompiler'
                        values 'ICC2022;intel/2022.2.0'
                    }
                    axis{
                        name 'HostSpace'
                        values 'SERIAL;-DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_ATOMICS_BYPASS=ON;-n 1', 'OPENMP;-DKokkos_ENABLE_OPENMP=ON;-n 10'
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
                            stage (' Test '){
                                steps {
                                    echo ' Testing '
                                    sh """
                                        module load boost/1.73.0
                                        module load ${GNU_COMPILER_MODULE}
                                        cd build_cpu_${GNU_COMPILER_NAME}_${CMAKE_HOST_NAME}_${SIMD_NAME}_${env.GIT_COMMIT}/tests/unit-tests
                                        srun -N 1 -t 00:20:00 ${HOST_RUN_FLAGS} --constraint=skylake bash -c 'export OMP_PROC_BIND=spread; export OMP_THREADS=places; ctest --verbose;'
                                    """
                                    echo ' Testing completed '
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
