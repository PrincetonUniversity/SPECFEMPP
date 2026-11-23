include: "meshfem2d.smk"


rule specfem2d_solver:
    input:
        setup=rules.specfem2d_setup.output,
        mesher=rules.specfem2d_mesher.output,
        cwd=rules.specfem2d_setup.output.cwd,
    output:
        solver="<cwd>/specfem2d_workdir/OUTPUT_FILES/output_solver.txt",
    shell:
        """
            cd {input.cwd}
            $SPECFEM2D_BINDIR/xspecfem2D > OUTPUT_FILES/output_solver.txt
        """


rule specfem2d_move_traces:
    input:
        solver=rules.specfem2d_solver.output.solver,
        cwd=rules.specfem2d_setup.output.cwd,
    output:
        trace_list="<cwd>/traces/trace_list.txt",
    localrule: True,
    run:
        import os
        from pathlib import Path
        trace_dir = os.path.join(input.cwd, "../traces")
        solver_outdir = os.path.join(input.cwd, "OUTPUT_FILES")
        os.makedirs(trace_dir, exist_ok=True)

        trace_list = []

        for trace_file in os.listdir(solver_outdir):
            trace_file_split = trace_file.split(".")
            if len(trace_file_split) == 4 and trace_file_split[-1].startswith("sem"):
                if trace_file_split[2] != "PRE":
                    trace_file_split[2] = "B" + trace_file_split[2][1:]
                trace_file_out = ".".join([trace_file_split[0], trace_file_split[1], "S2", trace_file_split[2], trace_file_split[3]])
                trace_list.append(trace_file_out + "\n")
                infile = os.path.join(solver_outdir, trace_file)
                outfile = os.path.join(trace_dir, trace_file_out)
                shell(f"mv {infile} {outfile}")

        with open(output.trace_list, "w") as f:
            f.writelines(trace_list)

        output_solver = Path(solver_outdir) / "output_solver.txt"
        output_solver.rename(output_solver.parent / "output_solver_store.txt")


rule clean:
    localrule: True
    shell:
        """
            rm -rf specfem2d_workdir
            rm -rf traces
            rm -f database.bin
        """
