
include: "variables.smk"

rule define_specfem_config:
    output:
        specfem_config = os.path.join(work_directory, "specfem_config.yaml")

    input:
        specfem_config = os.path.join(snakefile_dir, "rules/templates/specfem_config.yaml"),

    params:
        mesh_database = os.path.join(mesh_output_folder, "database.bin"),
        source_file = os.path.join(snakefile_dir, "rules/templates/sources.yaml"),
        stations_file = os.path.join(mesh_output_folder, "STATIONS")
        work_directory = f"{work_directory}"

    localrule:
        True

    run:
        with open(input.specfem_config, "r") as f:
            template = f.read()

        content = template.format(mesh_database=params.mesh_database,
                                  source_file=params.source_file,
                                  stations_file=params.stations_file,
                                  work_directory=params.work_directory)

        with open(output.specfem_config, "w") as f:
            f.write(content)

rule run_simulation:
    input:
        specfem_config = os.path.join(work_directory, "specfem_config.yaml"),
        mesh_database = os.path.join(mesh_output_folder, "database.bin"),
        stations = os.path.join(mesh_output_folder, "STATIONS")

    params:
        work_directory = f"{work_directory}"

    output:
        log = os.path.join(work_directory, "output.log")

    resources:
        nodes=1,
        tasks=1,
        cpus_per_task=1,
        runtime=10,
        constraint="skylake"

    shell:
        '''
            module purge
            module load boost/1.73.0
            mkdir -p {params.work_directory}/seismograms
            echo "Hostname: $(hostname)" > {output.log}
            specfem2d -p {input.specfem_config} >> {output.log}
        '''
