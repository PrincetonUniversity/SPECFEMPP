# SPECFEM++ mesher rules for the injection fixtures, taken from
# displacement_tests/Newmark/meshfem3d.smk without its Fortran reference rules,
# so only SPECFEMPP_BINDIR is needed.
#
# A mesh Snakefile sets `cores` (the MPI ranks the mesh is partitioned for)
# before including this file; its Mesh_Par_file must have
# NPROC_XI * NPROC_ETA == cores.

import os

envvars:
    "SPECFEMPP_BINDIR",

pathvars:
    cwd=os.getcwd()

cores = globals().get("cores", 1)
specfempp_mesh_files = ["interface1.txt", "interfaces.txt", "Mesh_Par_file"]

if cores > 1:
    _mesher_db = [
        f"<cwd>/workdir/OUTPUT_FILES/Database/proc_{rank}.bin"
        for rank in range(cores)
    ]
    _final_db = [f"<cwd>/database/proc_{rank}.bin" for rank in range(cores)]
else:
    _mesher_db = "<cwd>/workdir/OUTPUT_FILES/Database.bin"
    _final_db = "<cwd>/database.bin"


rule specfempp_setup:
    input:
        mesh_files=[f"<cwd>/provenance/specfempp/{f}" for f in specfempp_mesh_files],
    output:
        mesh_files=[f"<cwd>/workdir/{f}" for f in specfempp_mesh_files],
        cwd=directory("<cwd>/workdir"),
    localrule: True,
    shell:
        """
            mkdir -p {output.cwd}
            for file in {input.mesh_files}; do cp "$file" {output.cwd}/$(basename "$file"); done
        """


rule specfempp_mesher:
    input:
        cwd=rules.specfempp_setup.output.cwd,
        mesh_files=rules.specfempp_setup.output.mesh_files,
    output:
        database=_mesher_db,
    params:
        launcher=(f"mpirun -np {cores} " if cores > 1 else ""),
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
        database=_final_db,
    run:
        import shutil

        ins = input.database if isinstance(input.database, list) else [input.database]
        outs = output.database if isinstance(output.database, list) else [output.database]
        for src, dst in zip(ins, outs):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
