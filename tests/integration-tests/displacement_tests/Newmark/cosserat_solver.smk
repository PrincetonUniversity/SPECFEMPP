include: "meshfem3d.smk"


rule link_traces:
    input:
        trace_list="<cwd>/traces_analytic/trace_list.txt",
    output:
        trace_list="<cwd>/traces/trace_list.txt",
    localrule: True,
    shell:
        """
        mkdir -p $(dirname {output.trace_list})
        cd $(dirname {output.trace_list})
        ln -sf ../traces_analytic/* .
        """


rule run_cosserat_solver:
    input:
        params="<cwd>/provenance/analytic/params.yaml",
    output:
        log="<cwd>/provenance/analytic/output/cosserat_solver.log",
        outdir=directory("<cwd>/provenance/analytic/output"),
    threads: 32
    shell:
        """
        if ! command -v cosserat-solver &> /dev/null; then
            echo "Error: cosserat-solver is not installed or not on PATH"
            exit 1
        fi
        cd $(dirname {input.params})
        cosserat-solver --yaml params.yaml --o output
        """


rule clean:
    localrule: True,
    shell:
        """
        rm -rf traces
        rm -rf traces_analytic
        rm -f database.bin
        rm -rf specfem3d_workdir
        rm -rf provenance/analytic/output
        """
