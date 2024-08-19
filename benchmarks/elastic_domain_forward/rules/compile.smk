
include: "variables.smk"

rule compile:
    output:
        directory(build_dir)

    localrule:
            True

    params:
        source_dir = os.path.join(snakefile_dir, "../..")

    shell:
        '''
            module purge
            module load boost/1.73.0
            mkdir -p {output}
            cd {output}
            cmake3 -S {params.source_dir} -B . \
                -D CMAKE_BUILD_TYPE=Release \
                -D BUILD_TESTS=ON \
                -D ENABLE_SIMD=ON \
                -D Kokkos_ARCH_NATIVE=ON \
                -D Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON \
                -D Kokkos_ENABLE_ATOMICS_BYPASS=ON
            cmake3 --build . -j 8
        '''
