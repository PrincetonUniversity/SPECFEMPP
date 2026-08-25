pipeline {
    agent {
        node {
            label 'della-rse_specfempp'
        }
    }
    environment {
        // Must match the context of the status this job already publishes, or the row
        // is added alongside the old one instead of replacing it. Point this at
        // 'sandbox/status-format' to rehearse against a throwaway commit.
        GH_CONTEXT = 'MPI GCC Compiler Checks'
        GH_REPO_URL = 'https://github.com/PrincetonUniversity/SPECFEMPP'
    }
    stages{
        stage(' Reset Reporting '){
            steps {
                // Workspaces are reused between builds, so stale per-cell status files
                // would otherwise be folded into this build's commit-status summary.
                sh 'rm -rf status junit && mkdir -p status junit'
            }
        }
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
                        values 'MPI_GCC416;-DSPECFEM_ENABLE_MPI=ON;openmpi/gcc/4.1.6;--ntasks=8 --ntasks-per-node=4 --nodes=2'
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
                            // Cell identity, derived once. CELL names files (unique per
                            // matrix cell, since all cells share one workspace); LABEL is
                            // the short human-facing form that reaches the GitHub status.
                            CELL = "mpi_${GNU_COMPILER_NAME}_${MPI_NAME}_${CMAKE_HOST_NAME}_${SIMD_NAME}"
                            LABEL = "${GNU_COMPILER_NAME}/${SIMD_NAME}"
                            BUILD_DIR = "build_${CELL}_${env.BUILD_TAG}"
                            INSTALL_DIR = "install_${CELL}_${env.BUILD_TAG}"
                            TEST_DIR = "/scratch/gpfs/TROMP/specfempp/jenkins/test_${CELL}_${env.BUILD_TAG}"
                        }
                        stages {
                            stage (' Configure '){
                                steps {
                                    // Written before anything can fail: post{failure} does
                                    // not run on ABORTED (cancelled job, Slurm timeout), so
                                    // without this a killed cell would leave no file at all
                                    // and silently drop out of the summary.
                                    writeFile file: "status/${CELL}", text: "STARTED ${LABEL}\n"
                                    echo "Configuring ${CMAKE_HOST_FLAGS} ${SIMD_FLAGS} ${MPI_FLAGS} with ${GNU_COMPILER_NAME} and ${MPI_NAME}"
                                    sh """
                                        module load cmake/3.30.8
                                        module load boost/1.85.0
                                        module load ${GNU_COMPILER_MODULE}
                                        module load ${MPI_MODULE}
                                        export CC=mpicc
                                        export CXX=mpicxx
                                        export FC=mpifort
                                        cmake3 -S . -B ${BUILD_DIR} \
                                          -DCMAKE_BUILD_TYPE=Release \
                                          -D CMAKE_INSTALL_PREFIX=${INSTALL_DIR}/bin \
                                          ${CMAKE_HOST_FLAGS} ${SIMD_FLAGS} ${MPI_FLAGS} \
                                          -D SPECFEM_BUILD_TESTS=ON \
                                          -D SPECFEM_BUILD_BENCHMARKS=OFF \
                                          -D SPECFEMPP_TEST_DIR=${TEST_DIR}
                                    """
                                }
                                post {
                                    failure {
                                        writeFile file: "status/${CELL}", text: "CONFIGURE ${LABEL}\n"
                                    }
                                }
                            }
                            stage (' Compile '){
                                // Split from Configure purely so the commit status can say
                                // which of the two failed; they were one sh block before.
                                steps {
                                    echo " Compiling ${BUILD_DIR} "
                                    sh """
                                        module load cmake/3.30.8
                                        module load boost/1.85.0
                                        module load ${GNU_COMPILER_MODULE}
                                        module load ${MPI_MODULE}
                                        cmake3 --build ${BUILD_DIR}
                                    """
                                    echo ' Build completed '
                                }
                                post {
                                    failure {
                                        writeFile file: "status/${CELL}", text: "COMPILE ${LABEL}\n"
                                    }
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
                                        cd ${TEST_DIR}
                                        salloc ${MPI_RUN_FLAGS} -t 00:15:00 --account rse \
                                            --constraint="intel" \
                                            bash -c 'export OMP_PROC_BIND=spread; \
                                            export OMP_PLACES=threads; \
                                            export OMP_NUM_THREADS=20; \
                                            hostname; \
                                            echo "ranks run on: \$SLURM_JOB_NODELIST"; \
                                            ctest -R MPI --output-on-failure --no-tests=error \
                                                  --output-junit junit.xml;'
                                    """
                                    echo ' Testing completed '
                                }
                                post {
                                    always {
                                        // post{always}, not a step: when ctest fails the sh
                                        // above aborts the block, and the counts are exactly
                                        // what we need in that case.
                                        sh "sh .jenkins/record_test_result.sh '${TEST_DIR}/junit.xml' 'junit/${CELL}.xml' '${LABEL}' 'status/${CELL}'"
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
            script {
                // The `|| echo` guarantees a parseable line even if the script is missing
                // or the workspace was never populated -- an exception thrown here would
                // leave the PR showing a stale status with no indication why.
                def result = sh(returnStdout: true, script:
                    'sh .jenkins/status_description.sh status 2>/dev/null ' +
                    '|| echo "FAILURE could not summarize results"').trim()
                def state = result.tokenize(' ')[0]
                if (!(state in ['SUCCESS', 'FAILURE'])) {
                    state = 'FAILURE'
                    result = 'FAILURE unexpected summary: ' + result
                }
                def desc = result.substring(state.length()).trim()
                echo "GitHub status -> ${state}: ${desc}"
                // githubNotify does not exist on this controller (older github-plugin), so
                // this uses the class-based step instead. It picks up the globally
                // configured GitHub server credentials, so no credentialsId is needed.
                step([$class: 'GitHubCommitStatusSetter',
                      contextSource: [$class: 'ManuallyEnteredCommitContextSource',
                                      context: env.GH_CONTEXT],
                      reposSource: [$class: 'ManuallyEnteredRepositorySource',
                                    url: env.GH_REPO_URL],
                      commitShaSource: [$class: 'BuildDataRevisionShaSource'],
                      statusResultSource: [$class: 'ConditionalStatusResultSource',
                          results: [[$class: 'AnyBuildResult', state: state, message: desc]]]
                ])
            }
        }
    }
}
