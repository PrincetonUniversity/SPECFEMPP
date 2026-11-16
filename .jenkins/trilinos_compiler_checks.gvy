// What each axis value means.

// Only 17.1.1-cpu-native-nompi is actually installed. The four
// trilinos/16.1.0-* modulefiles still load, but their prefixes under
// /home/TROMP/source/Trilinos/install/ were deleted, so they are
// deliberately not listed. Add cells back once they are rebuilt.
def trilinos_module = [TRILINOS1711: 'trilinos/17.1.1-cpu-native-nompi']
// The host space is deliberately NOT an axis. Kokkos comes from the
// Trilinos module via find_package (cmake/kokkos.cmake) rather than
// being built here, and that install is Serial-only, so -DKokkos_ENABLE_*
// and -DKokkos_ARCH_* would be unused variables. SPECFEM_ENABLE_SIMD is
// ours and does vary. The compiler is not an axis either: Trilinos was
// built with /opt/rh/gcc-toolset-14, so SPECFEM++ must match it.
def simd_flags = [
    SIMD_NONE  : '-DSPECFEM_ENABLE_SIMD=OFF',
    SIMD_NATIVE: '-DSPECFEM_ENABLE_SIMD=ON'
]

pipeline {
    agent {
        node {
            label 'della-rse_specfempp'
        }
    }
    stages {
        stage('Reset Reporting') {
            steps {
                // Every matrix cell shares one workspace and workspaces are reused between
                // builds, so a junit/ left by an earlier build would be archived as if it
                // belonged to this one.
                sh 'rm -rf junit && mkdir -p junit'
            }
        }
        stage('Trilinos Implicit Solver Check') {
            matrix {
                axes {
                    axis {
                        name 'TrilinosModule'
                        values 'TRILINOS1711'
                    }
                    axis {
                        name 'SIMD'
                        values 'SIMD_NONE', 'SIMD_NATIVE'
                    }
                }
                // Read by .jenkins/cell.sh
                environment {
                    CELL            = "trilinos_${TrilinosModule}_${SIMD}"
                    // gcc-toolset/14 is loaded after Trilinos because the Trilinos
                    // module loads openblas/0.3, which messes with the
                    // LD_LIBRARY_PATH for gcc-toolset/14.
                    MODULES         = "${trilinos_module[TrilinosModule]} gcc-toolset/14"
                    LD_LIBRARY_PATH_PREPEND = '/opt/rh/gcc-toolset-14/root/usr/lib64'
                    CMAKE_FLAGS     = "${simd_flags[SIMD]} -DSPECFEM_ENABLE_TRILINOS=ON -DSPECFEM_ENABLE_DOUBLE_PRECISION=OFF"
                    // --constraint=intel because the module's Kokkos was built
                    // Kokkos_ARCH NATIVE, which resolved to AVX512XEON.
                    SLURM_FLAGS     = '-n 1 -c 4 -t 00:15:00 --constraint=intel'
                    CTEST_FLAGS     = '-L TRILINOS -j 4'
                    // This Kokkos is Serial-only, so the only OpenMP consumer is
                    // OpenBLAS, and letting it expand would oversubscribe the 4
                    // cores that -j 4 already uses.
                    OMP_NUM_THREADS = '1'
                }
                stages {
                    stage('Build') {
                        steps {
                            sh '.jenkins/cell.sh build'
                        }
                    }
                    stage('Test') {
                        steps {
                            sh '.jenkins/cell.sh test'
                        }
                    }
                }
                post {
                    always {
                        sh '.jenkins/cell.sh clean'
                    }
                }
            }
        }
    }
    post {
        always {
            junit testResults: 'junit/*.xml',
                  allowEmptyResults: true,
                  skipPublishingChecks: true
            archiveArtifacts artifacts: 'junit/*.xml', allowEmptyArchive: true
        }
    }
}
