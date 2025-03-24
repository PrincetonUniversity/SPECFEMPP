# Electromagnetic wave propagration of the trasnverse electric mode without damping

This example simulates EM wave propagation through an electromagnetic medium.
The example is based on the GPR TM example from the fortran version of SPECFEM2D.

Some important differences for this example are that it was adapted prior to
the implementation of damping and the transverse magnetic mode. In other words,
we simulate the domain using the transverse electric (TE) mode and no damping.
Importantly, Stacey boundary conditions have not yet been implemented for EM
domains meaning the domain is reflective at all boundaries.

For the original publication of this example please see: [Morency,
2020](https://academic.oup.com/gji/article/220/2/951/5625073).

## Running the examples

To run the examples, you first need to install uv following these
[instructions](https://docs.astral.sh/uv/getting-started/installation). Once
you've done so, you can install the dependencies for the examples by running the
following command in the current directory:

```bash
# verify uv is installed
uv --version

# install dependencies
uv sync --group examples

```

After installing the dependencies, you can run the examples by running the
following command within the example directory you want to run:

```bash

# run the example
uv run snakemake -j 1

# or to run the example on a slurm cluster
uv run snakemake --executor slurm -j 1

```

## Cleaning up

To clean up the example directory, you can run the following command in the
directory of the example you want to clean up:

```bash

# clean up the example
uv run snakemake clean -j 1

```
