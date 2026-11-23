envvars:
    "SPECFEM2D_BINDIR",
    "SPECFEMPP_BINDIR",

pathvars:
    cwd=os.getcwd()


rule specfem2d_setup:
    input:
        par_file="<cwd>/provenance/Par_file",
        topography="<cwd>/provenance/topography.dat",
    output:
        par_file="<cwd>/specfem2d_workdir/DATA/Par_file",
        topography="<cwd>/specfem2d_workdir/DATA/topography.dat",
        cwd=directory("<cwd>/specfem2d_workdir"),
    localrule: True,
    shell:
        """
            cp {input.par_file} {output.par_file}
            cp {input.topography} {output.topography}
        """


rule specfem2d_mesher:
    input:
        setup=rules.specfem2d_setup.output,
        cwd=rules.specfem2d_setup.output.cwd,
        source="<cwd>/provenance/SOURCE",
    output:
        database="<cwd>/specfem2d_workdir/OUTPUT_FILES/Database00000.bin",
        stations="<cwd>/specfem2d_workdir/DATA/STATIONS",
        mesher="<cwd>/specfem2d_workdir/OUTPUT_FILES/output_mesher.txt",
        source="<cwd>/specfem2d_workdir/DATA/SOURCE",
    shell:
        """
            cp {input.source} {output.source}
            cd {input.cwd}
            $SPECFEM2D_BINDIR/xmeshfem2D > OUTPUT_FILES/output_mesher.txt
        """


rule specfempp_mesher:
    input:
        setup=rules.specfem2d_setup.output,
        cwd=rules.specfem2d_setup.output.cwd,
    output:
        database="<cwd>/specfem2d_workdir/OUTPUT_FILES/database.bin",
    shell:
        """
            cd {input.cwd}/DATA
            $SPECFEMPP_BINDIR/xmeshfem2D -p Par_file
        """

rule specfempp_move_database:
    input:
        database=rules.specfempp_mesher.output.database,
    output:
        database="<cwd>/database.bin",
    shell:
        """
            mv {input.database} {output.database}
        """
