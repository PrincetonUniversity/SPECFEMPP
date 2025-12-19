include: "meshfem3d.smk"


rule specfem3d_solver:
    input:
        setup=rules.specfem3d_setup.output,
        mesher=rules.specfem3d_mesher.output,
        database=rules.specfem3d_generate_database.output.databases,
        cwd=rules.specfem3d_setup.output.cwd,
    output:
        solver="<cwd>/specfem3d_workdir/fortran/OUTPUT_FILES/output_solver.txt",
    shell:
        """
            cd {input.cwd}
            mpirun -n 1 $SPECFEM3D_BINDIR/xspecfem3D
        """


rule specfem3d_move_traces:
    input:
        solver=rules.specfem3d_solver.output.solver,
        cwd=rules.specfem3d_setup.output.cwd,
    output:
        trace_list="<cwd>/traces/trace_list.txt",
    localrule: True,
    run:
        import os
        trace_dir = os.path.join(input.cwd, "../../traces")
        solver_outdir = os.path.join(input.cwd, "OUTPUT_FILES")
        os.makedirs(trace_dir, exist_ok=True)

        trace_list = []

        for trace_file in os.listdir(solver_outdir):
            trace_file_split = trace_file.split(".")
            if len(trace_file_split) == 4 and trace_file_split[-1].startswith("sem"):
                trace_file_out = ".".join([trace_file_split[0], trace_file_split[1], "S3", trace_file_split[2], trace_file_split[3]])
                trace_list.append(trace_file_out + "\n")
                infile = os.path.join(solver_outdir, trace_file)
                outfile = os.path.join(trace_dir, trace_file_out)
                shell(f"mv {infile} {outfile}")

        with open(output.trace_list, "w") as f:
            f.writelines(trace_list)

        os.remove(os.path.join(solver_outdir, "output_solver.txt"))


rule clean:
    localrule: True
    shell:
        """
            rm -rf specfem3d_workdir
            rm -rf traces
            rm -f mesh_database.bin
        """
