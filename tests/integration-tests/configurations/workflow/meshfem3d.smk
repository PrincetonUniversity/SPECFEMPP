import glob
import os
import re

envvars:
    "SPECFEM3D_BINDIR",
    "SPECFEMPP_BINDIR",

pathvars:
    cwd=os.getcwd()

# Number of MPI ranks the SPECFEM++ mesher is partitioned for. Defaults to 1
# (serial). An MPI test Snakefile sets `cores = <nproc>` BEFORE including this
# module; the matching Mesh_Par_file must declare NPROC_XI * NPROC_ETA == cores.
cores = globals().get("cores", 1)

# SPECFEM++ partitioned databases follow specfem::MPI::format_proc_filename():
# "<dir>/<stem>.<ext>" -> "<dir>/<stem>/proc_N.<ext>" when size > 1, unchanged
# when size == 1. The mesher writes under LOCAL_PATH (./OUTPUT_FILES/), and the
# final per-test layout must match `mesh-database: .../database.bin` in the
# specfem_config, i.e. ".../database/proc_N.bin".
if cores > 1:
    _specfempp_mesher_db = [
        f"<cwd>/specfem3d_workdir/specfempp/OUTPUT_FILES/Database/proc_{rank}.bin"
        for rank in range(cores)
    ]
    _specfempp_final_db = [
        f"<cwd>/database/proc_{rank}.bin" for rank in range(cores)
    ]
else:
    _specfempp_mesher_db = "<cwd>/specfem3d_workdir/specfempp/OUTPUT_FILES/Database.bin"
    _specfempp_final_db = "<cwd>/database.bin"

# Fortran reference databases are written one file per rank when NPROC > 1
# (proc<6-digit>_*). The Fortran provenance (Par_file NPROC, Mesh_Par_file
# NPROC_XI*NPROC_ETA) must equal `cores`. The reference traces themselves are
# partition-independent.
if cores > 1:
    _fortran_mesher_db = [
        f"<cwd>/specfem3d_workdir/fortran/DATABASES_MPI/proc{rank:06d}_Database"
        for rank in range(cores)
    ]
    _fortran_external_mesh = [
        f"<cwd>/specfem3d_workdir/fortran/DATABASES_MPI/proc{rank:06d}_external_mesh.bin"
        for rank in range(cores)
    ]
else:
    _fortran_mesher_db = "<cwd>/specfem3d_workdir/fortran/DATABASES_MPI/proc000000_Database"
    _fortran_external_mesh = [
        f"<cwd>/specfem3d_workdir/fortran/DATABASES_MPI/proc000000_{parameter}.bin"
        for parameter in ["external_mesh"]
    ]

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
        database=_fortran_mesher_db,
        mesher="<cwd>/specfem3d_workdir/fortran/OUTPUT_FILES/output_meshfem3D.txt",
        # source="<cwd>/specfem3d_workdir/fortran/DATA/" + source_file,
        # stations="<cwd>/specfem3d_workdir/fortran/DATA/STATIONS",
    localrule: True,
    params:
        launcher=f"mpirun -n {cores}",
    shell:
        """
            cp -f {input.source} {input.cwd}/DATA/$(basename {input.source})
            cp -f {input.stations} {input.cwd}/DATA/STATIONS
            cd {input.cwd}
            echo "Running xmeshfem3D"
            mkdir -p OUTPUT_FILES
            mkdir -p DATABASES_MPI
            {params.launcher} $SPECFEM3D_BINDIR/xmeshfem3D -p {input.mesh_par_file}
        """

rule specfem3d_generate_database:
    input:
        mesher=rules.specfem3d_mesher.output.mesher,
        cwd=rules.specfem3d_setup.output.cwd,
        mesh_database=rules.specfem3d_mesher.output.database,
    output:
        databases=_fortran_external_mesh,
    params:
        launcher=f"mpirun -n {cores}",
    shell:
        """
            cd {input.cwd}
            echo "Generating database files"
            {params.launcher} $SPECFEM3D_BINDIR/xgenerate_databases
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
        database=_specfempp_mesher_db,
    params:
        # Empty for serial; "mpirun -np <cores> " for MPI partitioning.
        launcher=(f"mpirun -np {cores} " if cores > 1 else ""),
        # The mesher writes per-rank files into OUTPUT_FILES/Database/ under MPI.
        predir=("mkdir -p OUTPUT_FILES/Database" if cores > 1 else "true"),
    shell:
        """
            cd {input.cwd}
            mkdir -p OUTPUT_FILES
            {params.predir}
            {params.launcher}$SPECFEMPP_BINDIR/xmeshfem3D -p Mesh_Par_file
        """

rule specfempp_move_database:
    input:
        database=rules.specfempp_mesher.output.database,
    output:
        database=_specfempp_final_db,
    run:
        import os
        import shutil

        ins = input.database if isinstance(input.database, list) else [input.database]
        outs = output.database if isinstance(output.database, list) else [output.database]
        for src, dst in zip(ins, outs):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
