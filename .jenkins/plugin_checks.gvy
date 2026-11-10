// End-to-end checks of wavefield injection with the plane-wave example plugin
// (tests/integration-tests/injection). Each configuration builds SPECFEM++,
// installs the plugin SDK, builds examples/plugins/plane_wave against it, and
// rebuilds SPECFEM++ with SPECFEM_INJECTION_TEST_PLUGIN pointing at the plugin,
// which registers the injection_e2e tests.
pipeline {
    agent {
        node {
            label 'della_rk9481'
        }
    }

    stages {
        stage (' Injection plugin checks '){
            matrix {
                axes {
                    axis{
                        name 'HostSpace'
                        values 'SERIAL;-DKokkos_ENABLE_SERIAL=ON;-n 1', 'OPENMP;-DKokkos_ENABLE_OPENMP=ON;-n 10'
                    }
                    axis{
                        name 'DeviceSpace'
                        values 'NONE;-DKokkos_ENABLE_CUDA=OFF;--constraint=broadwell', 'CUDA_AMPERE80;-DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON;--gres=gpu:1 --constraint=a100'
                    }
                }

                stages {
                    stage ('Build and Clean '){
                        environment {
                            CUDA_MODULE='cudatoolkit/12.2'
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
                            HOST_RUN_FLAGS = """${sh(
                                                    returnStdout: true,
                                                    script: 'cut -d";" -f3 <<<"${HostSpace}"'
                                                ).trim()}"""
                            DEVICE_RUN_FLAGS = """${sh(
                                                    returnStdout: true,
                                                    script: 'cut -d";" -f3 <<<"${DeviceSpace}"'
                                                ).trim()}"""
                        }
                        stages {
                            stage (' Build '){
                                steps {
                                    echo "Building ${CMAKE_HOST_FLAGS} ${CMAKE_DEVICE_FLAGS} with the plane-wave plugin"
                                    sh """
                                        module load boost/1.73.0
                                        module load ${CUDA_MODULE}
                                        cmake3 -S . -B build_plugin_${CMAKE_HOST_NAME}_${CMAKE_DEVICE_NAME}_${env.GIT_COMMIT} -DCMAKE_BUILD_TYPE=Release ${CMAKE_HOST_FLAGS} ${CMAKE_DEVICE_FLAGS} -DSPECFEM_BUILD_TESTS=ON -DSPECFEM_BUILD_BENCHMARKS=OFF
                                        cmake3 --install build_plugin_${CMAKE_HOST_NAME}_${CMAKE_DEVICE_NAME}_${env.GIT_COMMIT} --component plugin_sdk --prefix build_plugin_${CMAKE_HOST_NAME}_${CMAKE_DEVICE_NAME}_${env.GIT_COMMIT}/plugin_sdk
                                        cmake3 -S examples/plugins/plane_wave -B build_plugin_${CMAKE_HOST_NAME}_${CMAKE_DEVICE_NAME}_${env.GIT_COMMIT}/plane_wave -DCMAKE_PREFIX_PATH=\$PWD/build_plugin_${CMAKE_HOST_NAME}_${CMAKE_DEVICE_NAME}_${env.GIT_COMMIT}/plugin_sdk -DCMAKE_BUILD_TYPE=Release
                                        cmake3 --build build_plugin_${CMAKE_HOST_NAME}_${CMAKE_DEVICE_NAME}_${env.GIT_COMMIT}/plane_wave
                                        cmake3 build_plugin_${CMAKE_HOST_NAME}_${CMAKE_DEVICE_NAME}_${env.GIT_COMMIT} -DSPECFEM_INJECTION_TEST_PLUGIN=\$PWD/build_plugin_${CMAKE_HOST_NAME}_${CMAKE_DEVICE_NAME}_${env.GIT_COMMIT}/plane_wave/libplane_wave.so
                                        cmake3 --build build_plugin_${CMAKE_HOST_NAME}_${CMAKE_DEVICE_NAME}_${env.GIT_COMMIT}
                                    """
                                    echo ' Build completed '
                                }
                            }
                            stage (' Run injection tests '){
                                steps {
                                    echo " Running the plugin's own tests and the end-to-end injection tests "
                                    sh """
                                        module load boost/1.73.0
                                        ctest --test-dir build_plugin_${CMAKE_HOST_NAME}_${CMAKE_DEVICE_NAME}_${env.GIT_COMMIT}/plane_wave --output-on-failure --no-tests=error
                                        cd build_plugin_${CMAKE_HOST_NAME}_${CMAKE_DEVICE_NAME}_${env.GIT_COMMIT}/tests
                                        srun -N 1 -t 00:30:00 ${HOST_RUN_FLAGS} ${DEVICE_RUN_FLAGS} bash -c 'export OMP_PROC_BIND=spread; export OMP_THREADS=places; ctest -L injection_e2e --output-on-failure --no-tests=error;'
                                    """
                                }
                            }
                        }
                        post {
                            always {
                                echo ' Cleaning '
                                sh "rm -rf build_plugin_${CMAKE_HOST_NAME}_${CMAKE_DEVICE_NAME}_${env.GIT_COMMIT}"
                            }
                        }
                    }
                }
            }
        }

        stage (' Injection plugin checks (MPI) '){
            environment {
                GNU_COMPILER_MODULE = 'gcc-toolset/14'
                MPI_MODULE = 'openmpi/gcc/4.1.6'
                BUILD_DIR = "build_plugin_mpi_${env.BUILD_TAG}"
                TEST_DIR = "/scratch/gpfs/TROMP/specfempp/jenkins/test_plugin_mpi_${env.BUILD_TAG}"
            }
            stages {
                stage (' Build '){
                    steps {
                        echo ' Building with MPI and the plane-wave plugin '
                        sh """
                            module load cmake/3.30.8
                            module load boost/1.85.0
                            module load ${GNU_COMPILER_MODULE}
                            module load ${MPI_MODULE}
                            export CC=mpicc
                            export CXX=mpicxx
                            export FC=mpifort
                            cmake3 -S . -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=Release \
                              -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_ATOMICS_BYPASS=ON \
                              -DSPECFEM_ENABLE_MPI=ON \
                              -DSPECFEM_BUILD_TESTS=ON \
                              -DSPECFEM_BUILD_BENCHMARKS=OFF \
                              -DSPECFEM_MPI_TEST_COMMAND=srun \
                              -DSPECFEMPP_TEST_DIR=${TEST_DIR}
                            cmake3 --install ${BUILD_DIR} --component plugin_sdk --prefix ${BUILD_DIR}/plugin_sdk
                            cmake3 -S examples/plugins/plane_wave -B ${BUILD_DIR}/plane_wave -DCMAKE_PREFIX_PATH=\$PWD/${BUILD_DIR}/plugin_sdk -DCMAKE_BUILD_TYPE=Release
                            cmake3 --build ${BUILD_DIR}/plane_wave
                            cmake3 ${BUILD_DIR} -DSPECFEM_INJECTION_TEST_PLUGIN=\$PWD/${BUILD_DIR}/plane_wave/libplane_wave.so
                            cmake3 --build ${BUILD_DIR}
                        """
                        echo ' Build completed '
                    }
                }
                stage (' Test '){
                    steps {
                        echo ' Running the end-to-end injection tests, serial and on 2 and 4 ranks '
                        sh """
                            module load cmake/3.30.8
                            module load boost/1.85.0
                            module load ${GNU_COMPILER_MODULE}
                            module load ${MPI_MODULE}
                            cd ${TEST_DIR}
                            # Same node exclusions as mpi_compiler_checks.gvy.
                            salloc --ntasks=4 --nodes=1-4 -t 00:30:00 --account rse \
                                --constraint="intel" \
                                --exclude="della-r3c[1-4]n[1-16],della-h17n1" \
                                bash -c 'hostname; \
                                echo "ranks run on: \$SLURM_JOB_NODELIST"; \
                                ctest -L injection_e2e --output-on-failure --no-tests=error;'
                        """
                        echo ' Testing completed '
                    }
                }
            }
            post {
                always {
                    echo 'Executing cleanup of build and test artifacts'
                    sh "rm -rf ${BUILD_DIR}"
                    sh "rm -rf ${TEST_DIR}"
                }
            }
        }
    }
}
