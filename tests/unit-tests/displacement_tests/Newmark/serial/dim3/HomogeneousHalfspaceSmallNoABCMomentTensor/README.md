# Homogeneous halfspace with Moment Tensor source, small model, no ABC

## Recreating the traces


Set the required environment variables and run the test as follows:

```bash
export SPECFEMPP_BINDIR=/home/lsawade/SPECFEMPP/bin/release
export SPECFEM3D_BINDIR=/home/lsawade/specfem3d/bin
```
Change your directory
```bash
cd tests/unit-tests/displacement_tests/Newmark/serial/dim3/HomogeneousHalfspaceSmallNoABCMomentTensor
```
Remove old traces and metadata if they exist
```bash
rm -rf .snakemake/ traces/* database.bin specfem3d_workdir/
```
run the snakemake workflow with the following command:
```bash
snakemake -j 1
```
