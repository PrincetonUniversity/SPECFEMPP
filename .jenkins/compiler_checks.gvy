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
        stage(' GNU Compilation Check '){
            matrix {
                axes {
                    axis{
                        name 'HostSpace'
                        values 'SERIAL;-DKokkos_ENABLE_SERIAL=ON', 'OPENMP;-DKokkos_ENABLE_OPENMP=ON'
                    }
                    axis{
                        name 'DeviceSpace'
                        values 'NONE;-DKokkos_ENABLE_CUDA=OFF', 'CUDA_AMPERE80;-DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON', 'CUDA_VOLTA70;-DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_VOLTA70=ON', 'CUDA_PASCAL60;-DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_PASCAL60=ON'
                    }
                }
                stages {
                    stage ('Build and Clean '){
                        environment {
                            CUDA_MODULE='cudatoolkit/11.7'
                            // CMAKE build flags
                            CMAKE_HOST_NAME = """${sh(
                                                    returnStdout: true,
                                                    script: 'cut -d";" -f1 <<<"${HostSpace}"'
                                                ).trim()}"""
                            CMAKE_HOST_FLAGS = """${sh(
                                                    returnStdout: true,
                                                    script: 'cut -d";" -f2 <<<"${HostSpace}"'
                                                ).trim()}"""
                            CMAKE_DEVICE_NAME = """${sh(
                                                    returnStdout: true,
                                                    script: 'cut -d";" -f1 <<<"${DeviceSpace}"'
                                                ).trim()}"""
                            CMAKE_DEVICE_FLAGS = """${sh(
                                                    returnStdout: true,
                                                    script: 'cut -d";" -f2 <<<"${DeviceSpace}"'
                                                ).trim()}"""
                        }
                        stages {
                            stage (' Build '){
                                steps {
                                    echo "Building ${CMAKE_HOST_FLAGS} ${CMAKE_DEVICE_FLAGS}"
                                    sh """
                                        module load boost/1.73.0
                                        module load ${CUDA_MODULE}
                                        cmake3 -S . -B build_GNU_${CMAKE_HOST_NAME}_${CMAKE_DEVICE_NAME} -DCMAKE_BUILD_TYPE=Release ${CMAKE_HOST_FLAGS} ${CMAKE_DEVICE_FLAGS}
                                        cmake3 --build build_GNU_${CMAKE_HOST_NAME}_${CMAKE_DEVICE_NAME}
                                    """
                                    echo ' Build completed '
                                }
                            }
                            stage (' Clean '){
                                steps {
                                    echo ' Cleaning '
                                    sh "rm -rf build_GNU_${CMAKE_HOST_NAME}_${CMAKE_DEVICE_NAME}"
                                }
                            }
                        }
                    }
                }
            }
        }
        stage(' Intel Compilation Check '){
            matrix {
                axes {
                    axis{
                        name 'HostSpace'
                        values 'SERIAL;-DKokkos_ENABLE_SERIAL=ON', 'OPENMP;-DKokkos_ENABLE_OPENMP=ON'
                    }
                    axis{
                        name 'DeviceSpace'
                        values 'NONE;-DKokkos_ENABLE_CUDA=OFF'
                    }
                }
                stages {
                    stage ('Build and Clean '){
                        environment {
                            // CMAKE build flags
                            CMAKE_HOST_NAME = """${sh(
                                                    returnStdout: true,
                                                    script: 'cut -d";" -f1 <<<"${HostSpace}"'
                                                ).trim()}"""
                            CMAKE_HOST_FLAGS = """${sh(
                                                    returnStdout: true,
                                                    script: 'cut -d";" -f2 <<<"${HostSpace}"'
                                                ).trim()}"""
                            CMAKE_DEVICE_NAME = """${sh(
                                                    returnStdout: true,
                                                    script: 'cut -d";" -f1 <<<"${DeviceSpace}"'
                                                ).trim()}"""
                            CMAKE_DEVICE_FLAGS = """${sh(
                                                    returnStdout: true,
                                                    script: 'cut -d";" -f2 <<<"${DeviceSpace}"'
                                                ).trim()}"""
                        }
                        stages {
                            stage (' Build '){
                                steps {
                                    echo "Building ${CMAKE_HOST_FLAGS} ${CMAKE_DEVICE_FLAGS}"
                                    sh """
                                        module load boost/1.73.0
                                        module load intel/2022.2.0
                                        export CC=icx
                                        export CXX=icpx
                                        cmake3 -S . -B build_INTEL_${CMAKE_HOST_NAME}_${CMAKE_DEVICE_NAME} -DCMAKE_BUILD_TYPE=Release ${CMAKE_HOST_FLAGS} ${CMAKE_DEVICE_FLAGS}
                                        cmake3 --build build_INTEL_${CMAKE_HOST_NAME}_${CMAKE_DEVICE_NAME}
                                    """
                                    echo ' Build completed '
                                }
                            }
                            stage (' Clean '){
                                steps {
                                    echo ' Cleaning '
                                    sh "rm -rf build_INTEL_${CMAKE_HOST_NAME}_${CMAKE_DEVICE_NAME}"
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
