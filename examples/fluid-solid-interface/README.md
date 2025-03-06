# Wave propagration through fluid-solid interface

This example creates the fluid-solid example with flat ocean bottom from
[Komatitsch et. al.](https://doi.org/10.1190/1.1444758). For a step-by-step
guide on this example, please refer to the
[documentation](https://specfem2d-kokkos.readthedocs.io/en/latest/cookbooks/dim2/fluid-solid-interface/index.html).

## Running the examples

To run the examples, you first need to install uv following these
[instructions](https://docs.astral.sh/uv/getting-started/installation). Once you've done
so, you can install the dependencies for the examples by running the following
command in the current directory:

```bash
# verify uv is installed
uv --version

# install dependencies
uv sync --extra examples

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
