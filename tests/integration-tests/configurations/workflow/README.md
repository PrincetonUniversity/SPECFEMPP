This Snakefile generates the trace files with provenance data.
Usage:
- Set environment variable to the folder containing Specfem2D binaries `xmeshfem2D` and `xspecfem2D`
```
export SPECFEM2D_BINDIR=<PATH_TO_SPECFEM2D_BINARIES>
```
- Run snakemake
```
snakemake -j 1
```
