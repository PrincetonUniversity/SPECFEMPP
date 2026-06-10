envvars:
    "SPECFEMPP_BINDIR",


pathvars:
    cwd=os.getcwd()


rule meshfem:
    input:
        cwd="<cwd>",
        par_file = "<cwd>/provenance/DATA/meshfem3D_files/Mesh_Par_file",
    output:
        database = "<cwd>/database.bin",
    shell:
        """
        cd {input.cwd}
        mkdir -p OUTPUT_FILES
        mkdir -p OUTPUT_FILES/DATABASES_MPI
        $SPECFEMPP_BINDIR/xmeshfem3D -p provenance/DATA/meshfem3D_files/Mesh_Par_file
        mv OUTPUT_FILES/DATABASES_MPI/Database.bin database.bin
        """

rule clean:
    shell:
        """
        rm -rf OUTPUT_FILES
        rm -f database.bin
        """
