import os


envvars:
    "SPECFEMPP_BINDIR",


pathvars:
    cwd=os.getcwd()


nproc = globals().get("nproc", 6)
databases = expand(
    "<cwd>/DATABASES_MPI/proc{rank:06d}_specfempp_database.bin",
    rank=range(nproc),
)


rule check_executable:
    input:
        mesher=os.path.join(os.environ["SPECFEMPP_BINDIR"], "xmeshfem3D_globe"),
    output:
        executable_checked="<cwd>/executable_checked.txt",
    shell:
        """
        if ! test -x {input.mesher}; then
            echo "Error: {input.mesher} not found or not executable."
            exit 1
        fi
        touch {output}
        """


rule meshfem3d_globe:
    input:
        executable_checked=rules.check_executable.output.executable_checked,
        mesher=rules.check_executable.input.mesher,
        par_file="<cwd>/provenance/DATA/Par_file",
        stations="<cwd>/provenance/DATA/STATIONS",
        source="<cwd>/provenance/DATA/CMTSOLUTION",
    output:
        databases=databases,
        log="<cwd>/OUTPUT_FILES/output_mesher.txt",
    resources:
        mpi="mpirun",
        tasks=nproc,
        cpus_per_task=1,
        runtime=1,
    shell:
        """
        cd <cwd>
        mkdir -p DATABASES_MPI OUTPUT_FILES
        rm -rf DATA
        cp -R provenance/DATA DATA
        {resources.mpi} -np {resources.tasks} {input.mesher} > OUTPUT_FILES/output_mesher.txt 2>&1
        rm -rf DATA
        """


rule clean:
    shell:
        """
        rm -rf <cwd>/DATABASES_MPI <cwd>/OUTPUT_FILES <cwd>/DATA <cwd>/.snakemake
        rm -f <cwd>/executable_checked.txt
        """
