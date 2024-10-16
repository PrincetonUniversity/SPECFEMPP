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
                                    echo "Building ${CMAKE_HOST_FLAGS} ${SIMD_FLAGS} with ${GNU_COMPILER_NAME}"
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
    }
}
