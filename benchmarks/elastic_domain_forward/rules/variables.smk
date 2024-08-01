
snakefile_dir = workflow.basedir
work_directory = os.path.join(snakefile_dir, "workdir_{nxmax}_{nzmax}_{repeat}")
build_dir = os.path.join(snakefile_dir, "build_elastic_domain_forward")
