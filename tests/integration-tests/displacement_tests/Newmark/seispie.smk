include: "meshfem2d.smk"


rule link_traces:
    input:
        trace_list="<cwd>/traces_fd/trace_list.txt",
    output:
        trace_list="<cwd>/traces/trace_list.txt",
    localrule: True,
    shell:
        """
        mkdir -p $(dirname {output.trace_list})
        cd $(dirname {output.trace_list})
        ln -sf ../traces_fd/* .
        """


rule run_seispie:
    input:
        config="<cwd>/provenance/seispie/config.ini",
        model="<cwd>/provenance/seispie/generate_model.py",
        sources="<cwd>/provenance/seispie/sources.dat",
        stations="<cwd>/provenance/seispie/stations.dat",
    output:
        ry="<cwd>/provenance/seispie/output/ry_000000.npy",
        iry="<cwd>/provenance/seispie/output/iry_000000.npy",
        cry="<cwd>/provenance/seispie/output/cry_000000.npy",
        ux="<cwd>/provenance/seispie/output/ux_000000.npy",
        uz="<cwd>/provenance/seispie/output/uz_000000.npy",
    shell:
        """
        if [ -z "$SEISPIE_DIR" ]; then
            echo "Error: SEISPIE_DIR environment variable is not set"
        fi
        cd $(dirname {input.model})
        mkdir -p model
        $SEISPIE_DIR/.venv/bin/python generate_model.py
        PYTHONPATH=$SEISPIE_DIR $SEISPIE_DIR/.venv/bin/python $SEISPIE_DIR/scripts/sprun --workdir=$PWD
        """


rule clean:
    localrule: True,
    shell:
        """
        rm -rf traces
        rm -rf traces_fd
        rm -f database.bin
        rm -rf specfem2d_workdir
        rm -rf provenance/seispie/output
        rm -rf provenance/seispie/model
        """
