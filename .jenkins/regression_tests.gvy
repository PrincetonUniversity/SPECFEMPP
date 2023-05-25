pipeline {
    agent {
        node {
            label 'della_rk9481'
        }
    }

    stages {
        stage (' Build and run PR branch '){

            stages {
                stage (' Build '){
                    stages {
                        stage (' Update git modules '){
                            steps {
                                echo ' Getting git submodules '
                                sh 'git submodule init'
                                sh 'git submodule update'
                            }
                        }

                        stage (' Build CPU '){
                            steps {
                                echo " Building SPECFEM "
                                sh """
                                    cmake -S . -B build_cpu -DCMAKE_BUILD_TYPE=Release
                                    cmake3 --build build_cpu
                                """
                            }
                        }

                        stage (' Build GPU '){
                            steps {
                                echo " Building SPECFEM "
                                sh """
                                    module load cudatoolkit/11.7
                                    cmake -S . -B build_gpu -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON -DKokkos_ENABLE_OPENMP=ON
                                    cmake3 --build build_gpu
                                """
                            }
                        }
                    }
                }

                stage (' Run regression tests '){
                    steps {
                        sh """
                            mkdir -p regression-tests/results
                            srun -N 1 -n 1 -t 00:10:00 --constraint=broadwell bash tests/regression-tests/run.sh -d cpu -i tests/regression-tests -e build_cpu/specfem2d -r regression-tests/results/PR-cpu.yaml
                            srun -N 1 -c 10 -t 00:10:00 --gres=gpu:1 --constraint=a100 bash tests/regression-tests/run.sh -d gpu -i tests/regression-tests -e build_gpu/specfem2d -r regression-tests/results/PR-gpu.yaml
                        """
                    }
                }

                stage ( ' clean '){
                    steps {
                        sh """
                            rm -rf build_cpu
                            rm -rf build_gpu
                        """
                    }
                }
            }
        }

        stage (' Build and Test main branch '){
            stages {
                stage (' Checkout main branch '){
                    steps {
                        checkout([$class: 'GitSCM',
                                branches: [[name: 'regression-testing']],
                                userRemoteConfigs: [[url: 'https://github.com/PrincetonUniversity/specfem2d_kokkos']]])
                    }
                }

                stage (' Build '){
                    stages {
                        stage (' Update git modules '){
                            steps {
                                echo ' Getting git submodules '
                                sh 'git submodule init'
                                sh 'git submodule update'
                            }
                        }

                        stage (' Build CPU '){
                            steps {
                                echo " Building SPECFEM "
                                sh """
                                    cmake -S . -B build_cpu -DCMAKE_BUILD_TYPE=Release
                                    cmake3 --build build_cpu
                                """
                            }
                        }

                        stage (' Build GPU '){
                            steps {
                                echo " Building SPECFEM "
                                sh """
                                    module load cudatoolkit/11.7
                                    cmake -S . -B build_gpu -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON -DKokkos_ENABLE_OPENMP=ON
                                    cmake3 --build build_gpu
                                """
                            }
                        }
                    }
                }

                stage (' Run regression tests '){
                    steps {
                        sh """
                            mkdir -p regression-tests/results
                            srun -N 1 -n 1 -t 00:10:00 --constraint=broadwell bash tests/regression-tests/run.sh -d cpu -i tests/regression-tests -e build_cpu/specfem2d -r regression-tests/results/main-cpu.yaml
                            srun -N 1 -c 10 -t 00:10:00 --gres=gpu:1 --constraint=a100 bash tests/regression-tests/run.sh -d gpu -i tests/regression-tests -e build_gpu/specfem2d -r regression-tests/results/main-gpu.yaml
                        """
                    }
                }

                stage ( ' clean '){
                    steps {
                        sh """
                            rm -rf build_cpu
                            rm -rf build_gpu
                        """
                    }
                }
            }
        }

        stage (' Compare results '){
            steps {
                sh """
                    cd build/tests/regression-tests
                    ./build/tests/regression-tests/compare_regression_tests --pr regression-results/results/PR-cpu.yaml --main regression-results/results/main-cpu.yaml && \
                    ./build/tests/regression-tests/compare_regression_tests --pr regression-results/results/PR-gpu.yaml --main regression-results/results/main-gpu.yaml
                """
            }
        }
    }
}
