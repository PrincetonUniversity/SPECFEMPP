# Homogeneous acoustic halfspace with vertical up force in the elastic medium, small model, no ABC

This test really checks whether the acoustic-elastic coupling is working correctly.

## Recreating the traces

Set the required environment variables and run the test as follows:

```bash
export SPECFEMPP_BINDIR=/home/lsawade/SPECFEMPP/bin/release
export SPECFEM3D_BINDIR=/home/lsawade/specfem3d/bin
```

Change your directory to `tests/unit-tests/displacement_tests/Newmark/serial/dim3/AcousticElasticForce` and
run the snakemake workflow with the following command:

```bash
snakemake -j 1
```
