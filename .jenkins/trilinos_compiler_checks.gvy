pipeline{
    agent {
        node {
            label 'della-rse_specfempp'
        }
    }
    stages{
        stage(' Reset Reporting '){
            steps {
                // Every matrix cell shares one workspace and workspaces are reused between
                // builds, so a junit/ left by an earlier build would be archived as if it
                // belonged to this one.
                sh 'rm -rf junit && mkdir -p junit'
            }
        }
        stage(' Trilinos Implicit Solver Check '){
            matrix {
                axes {
                    // Only 17.1.1-cpu-native-nompi is actually installed. The four
                    // trilinos/16.1.0-* modulefiles still load, but their prefixes under
                    // /home/TROMP/source/Trilinos/install/ were deleted, so they are
                    // deliberately not listed. Add cells back once they are rebuilt.
                    axis{
                        name 'TrilinosModule'
                        values 'TRILINOS1711;trilinos/17.1.1-cpu-native-nompi'
                    }
                    // The host space is deliberately NOT an axis. Kokkos comes from the
                    // Trilinos module via find_package (cmake/kokkos.cmake) rather than
                    // being built here, and that install is Serial-only, so -DKokkos_ENABLE_*
                    // and -DKokkos_ARCH_* would be unused variables. SPECFEM_ENABLE_SIMD is
                    // ours and does vary. The compiler is not an axis either: Trilinos was
                    // built with /opt/rh/gcc-toolset-14, so SPECFEM++ must match it.
                    axis{
                        name 'SIMD'
                        values 'SIMD_NONE;-DSPECFEM_ENABLE_SIMD=OFF', 'SIMD_NATIVE;-DSPECFEM_ENABLE_SIMD=ON'
                    }
                }
                stages {
                    stage ('Build and Clean '){
                        environment {
                            // CMAKE build flags
                            TRILINOS_NAME = """${sh(
                                                    returnStdout: true,
                                                    script: 'cut -d";" -f1 <<<"${TrilinosModule}"'
                                                ).trim()}"""
                            TRILINOS_MODULE = """${sh(
                                                    returnStdout: true,
                                                    script: 'cut -d";" -f2 <<<"${TrilinosModule}"'
                                                ).trim()}"""
                            SIMD_NAME = """${sh(
                                                    returnStdout: true,
                                                    script: 'cut -d";" -f1 <<<"${SIMD}"'
                                                ).trim()}"""
                            SIMD_FLAGS = """${sh(
                                                    returnStdout: true,
                                                    script: 'cut -d";" -f2 <<<"${SIMD}"'
                                                ).trim()}"""
                            // Cell identity, derived once. CELL is unique per matrix cell,
                            // which is what keeps the shared workspace from colliding.
                            CELL = "trilinos_${TRILINOS_NAME}_${SIMD_NAME}"
                            BUILD_DIR = "build_${CELL}_${env.BUILD_TAG}"
                            INSTALL_DIR = "install_${CELL}_${env.BUILD_TAG}"
                            TEST_DIR = "/scratch/gpfs/TROMP/specfempp/jenkins/test_${CELL}_${env.BUILD_TAG}"
                        }
                        stages {
                            stage (' Build '){
                                steps {
                                    echo "Building ${SIMD_FLAGS} against ${TRILINOS_MODULE}"
                                    sh """
                                        module purge
                                        module load cmake/3.30.8
                                        module load boost/1.85.0
                                        module load ${TRILINOS_MODULE}
                                        # needs to be loaded after Trilinos because Trilinos module loads openblas/0.3
                                        # which messes with the LD_LIBRARY_PATH for gcc-toolset/14.
                                        module load gcc-toolset/14
                                        cmake3 -S . -B ${BUILD_DIR} \
                                          -DCMAKE_BUILD_TYPE=Release \
                                          -D CMAKE_INSTALL_PREFIX=${INSTALL_DIR}/bin \
                                          ${SIMD_FLAGS} \
                                          -D SPECFEM_ENABLE_TRILINOS=ON \
                                          -D SPECFEM_ENABLE_DOUBLE_PRECISION=OFF \
                                          -D SPECFEM_BUILD_TESTS=ON \
                                          -D SPECFEM_BUILD_BENCHMARKS=OFF \
                                          -D SPECFEMPP_TEST_DIR=${TEST_DIR}
                                        cmake3 --build ${BUILD_DIR}
                                    """
                                    echo ' Build completed '
                                }
                            }
                            stage (' Test '){
                                steps {
                                    echo ' Running Trilinos tests with ctest '
                                    sh """
                                        module purge
                                        module load cmake/3.30.8
                                        module load boost/1.85.0
                                        module load ${TRILINOS_MODULE}
                                        module load gcc-toolset/14
                                        export LD_LIBRARY_PATH=/opt/rh/gcc-toolset-14/root/usr/lib64:\$LD_LIBRARY_PATH
                                        # --constraint=intel because the module's Kokkos was built
                                        # Kokkos_ARCH NATIVE, which resolved to AVX512XEON.
                                        # OMP_NUM_THREADS=1: this Kokkos is Serial-only, so the
                                        # only OpenMP consumer is OpenBLAS, and letting it expand
                                        # would oversubscribe the 4 cores that -j 4 already uses.
                                        cd ${TEST_DIR} && \
                                        srun -t 00:15:00 --account rse -n 1 -c 4 \
                                            --constraint="intel" \
                                            bash -c 'export OMP_PROC_BIND=spread; \
                                            export OMP_PLACES=threads; \
                                            export OMP_NUM_THREADS=1; \
                                            hostname; \
                                            ctest -L TRILINOS -j 4 --output-on-failure \
                                                  --no-tests=error --output-junit junit.xml;'
                                    """
                                    echo ' Testing completed '
                                }
                                post {
                                    always {
                                        // post{always}, not a step: a ctest failure aborts the sh
                                        // block above, and that is exactly the run worth
                                        // publishing. `|| true` because ctest writes junit.xml
                                        // only at the end, so a Slurm timeout leaves no file.
                                        // This runs before the outer stage's cleanup, so TEST_DIR
                                        // still exists here.
                                        sh "cp '${TEST_DIR}/junit.xml' 'junit/${CELL}.xml' || true"
                                        junit testResults: "junit/${CELL}.xml",
                                              allowEmptyResults: true,
                                              skipPublishingChecks: true
                                    }
                                }
                            }
                        }
                        post {
                            always {
                                echo 'Executing cleanup of build and test artifacts'
                                sh "rm -rf ${BUILD_DIR}"
                                sh "rm -rf ${INSTALL_DIR}"
                                sh "rm -rf ${TEST_DIR}"
                            }
                        }
                    }
                }
            }
        }
    }
    post {
        always {
            archiveArtifacts artifacts: 'junit/*.xml', allowEmptyArchive: true
        }
    }
}
