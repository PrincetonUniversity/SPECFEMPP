import glob
import os
import re

envvars:
    "SPECFEM3D_BINDIR",
    "SPECFEMPP_BINDIR",

pathvars:
    cwd=os.getcwd()

rule specfem3d_setup:
    input:
        par_file="<cwd>/provenance/fortran/DATA/Par_file",
        mesh_files=[f"<cwd>/provenance/fortran/DATA/{f}" for f in fortran_mesh_files],
    output:
        par_file="<cwd>/specfem3d_workdir/fortran/DATA/Par_file",
        mesh_par_file="<cwd>/specfem3d_workdir/fortran/DATA/meshfem3D_files/Mesh_Par_file",
        mesh_files=[f"<cwd>/specfem3d_workdir/fortran/DATA/{f}" for f in fortran_mesh_files],
        cwd=directory("<cwd>/specfem3d_workdir/fortran"),
    localrule: True,
    shell:
        """
            mkdir -p {output.cwd}/fortran/DATA/meshfem3D_files
            mkdir -p {output.cwd}/fortran/OUTPUT_FILES
            cp {input.par_file} {output.par_file}
            for file in {input.mesh_files}; do cp "$file" {output.cwd}/DATA/meshfem3D_files/$(basename "$file"); done
        """


rule specfem3d_mesher:
    input:
        setup=rules.specfem3d_setup.output,
        cwd=rules.specfem3d_setup.output.cwd,
        mesh_par_file=rules.specfem3d_setup.output.mesh_par_file,
        mesh_files=rules.specfem3d_setup.output.mesh_files,
        source=ancient("<cwd>/provenance/fortran/DATA/" + source_file),
        stations=ancient("<cwd>/provenance/fortran/DATA/STATIONS"),
    output:
        database="<cwd>/specfem3d_workdir/fortran/DATABASES_MPI/proc000000_Database",
        mesher="<cwd>/specfem3d_workdir/fortran/OUTPUT_FILES/output_meshfem3D.txt",
        # source="<cwd>/specfem3d_workdir/fortran/DATA/" + source_file,
        # stations="<cwd>/specfem3d_workdir/fortran/DATA/STATIONS",
    localrule: True,
    shell:
        """
            cp -f {input.source} {input.cwd}/DATA/$(basename {input.source})
            cp -f {input.stations} {input.cwd}/DATA/STATIONS
            cd {input.cwd}
            echo "Running xmeshfem3D"
            mkdir -p OUTPUT_FILES
            mkdir -p DATABASES_MPI
            mpirun -n 1 $SPECFEM3D_BINDIR/xmeshfem3D -p {input.mesh_par_file}
        """

rule specfem3d_generate_database:
    input:
        mesher=rules.specfem3d_mesher.output.mesher,
        cwd=rules.specfem3d_setup.output.cwd,
        mesh_database=rules.specfem3d_mesher.output.database,
    output:
        databases=[f"<cwd>/specfem3d_workdir/fortran/DATABASES_MPI/proc000000_{parameter}.bin" for parameter in ["external_mesh"]] #, "ibool", "qkappa", "qmu", "rho", "vp", "vs", "x", "y", "z"]],
    shell:
        """
            cd {input.cwd}
            echo "Generating database files"
            mpirun -n 1 $SPECFEM3D_BINDIR/xgenerate_databases
        """

rule specfempp_setup:
    input:
        mesh_files=[f"<cwd>/provenance/specfempp/{f}" for f in specfempp_mesh_files],
    output:
        mesh_par_file="<cwd>/specfem3d_workdir/specfempp/Mesh_Par_file",
        mesh_files=[f"<cwd>/specfem3d_workdir/specfempp/{f}" for f in specfempp_mesh_files],
        cwd=directory("<cwd>/specfem3d_workdir/specfempp"),
    localrule: True,
    shell:
        """
            mkdir -p {output.cwd}/specfempp
            for file in {input.mesh_files}; do cp "$file" {output.cwd}/$(basename "$file"); done
        """

rule specfempp_mesher:
    input:
        setup=rules.specfempp_setup.output,
        cwd=rules.specfempp_setup.output.cwd,
        mesh_par_file=rules.specfempp_setup.output.mesh_par_file,
        mesh_files=rules.specfempp_setup.output.mesh_files,
    output:
        database="<cwd>/specfem3d_workdir/specfempp/OUTPUT_FILES/Database.bin",
    shell:
        """
            cd {input.cwd}
            mkdir -p OUTPUT_FILES
            $SPECFEMPP_BINDIR/xmeshfem3D -p Mesh_Par_file
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
