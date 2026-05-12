# Homogeneous acoustic halfspace with isotropic explosion in the elastic medium, small model, no ABC

This test really checks whether the acoustic-elastic coupling is working correctly.

## Expected arrivals at the receiver right above the source

Note that since it is an explosive source we expect to see only P wave arrivals in the traces. and very little to no S wave energy.

**Z component traces**

- 6.6s - direct P wave
- 26.6s the reflected wave from the acoustic-elastic boundary
- 40.0s the reflected wave from the bottom free surface
- [53.3s the reflected wave from the acoustic free surface of the model] Cut off right before so that the reflected wave from the acoustic free surface of the model is not visible in the traces.

**X component traces**

- 6.6s - direct P wave (barely noticable in the X component)
- ~26.6s the reflected wave from the acoustic-elastic boundary~ (not visible in the X component)
- ~40.0s the reflected wave from the bottom free surface~ (not visible in the X component)
- ~53.3s the reflected wave from the acoustic free surface of the model~ (not visible in the X component)

## Recreating the traces

Set the required environment variables and run the test as follows:

```bash
export SPECFEMPP_BINDIR=/home/lsawade/SPECFEMPP/bin/release
export SPECFEM3D_BINDIR=/home/lsawade/specfem3d/bin
```

Change your directory to `tests/unit-tests/displacement_tests/Newmark/serial/dim3/AcousticElasticForceSource` and
run the snakemake workflow with the following command:

```bash
snakemake -j 1
```
