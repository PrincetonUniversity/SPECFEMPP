
include: "variables.smk"

mesh_output_folder = os.path.join(work_directory, "mesh/OUTPUT_FILES")

rule define_par_file:
    output:
        par_file = os.path.join(work_directory, "mesh.par"),
        interfaces = os.path.join(work_directory, "topography.dat")

    input:
        par_file = os.path.join(snakefile_dir, "rules/templates/Par_file"),
        interfaces = os.path.join(snakefile_dir, "rules/templates/topography.dat")

    params:
        mesh_output_folder = f"{mesh_output_folder}"

    localrule:
        True

    run:
        with open(input.par_file, "r") as f:
            template = f.read()

        content = template.format(output_folder=params.mesh_output_folder,
                                  interfacesfile = output.interfaces,
                                  nxmax = wildcards.nxmax,
                                  nzmax = wildcards.nzmax)

        with open(output.par_file, "w") as f:
            f.write(content)

        with open(input.interfaces, "r") as f:
            template = f.read()

        content = template.format(nzmax = wildcards.nzmax)

        with open(output.interfaces, "w") as f:
            f.write(content)

rule generate_mesh:
    input:
        par_file = os.path.join(work_directory, "mesh.par"),
        interfaces = os.path.join(work_directory, "topography.dat")

    output:
        database = os.path.join(mesh_output_folder, "database.bin"),
        stations = os.path.join(mesh_output_folder, "STATIONS")

    localrule:
        True

    shell:
        '''
            xmeshfem2D -p {input.par_file}
        '''
