# Wave propagation through homogeneous medium with no interfaces

In this example we simulate wave propagation through a 2-dimensional homogeneous medium. For a step-by-step guide on this example, please refer to the [documentation](https://specfem2d-kokkos.readthedocs.io/en/latest/cookbooks/dim2/homogeneous-medium/index.html).

## Running the examples

To run the examples, you first need to install poetry following these [instructions](https://python-poetry.org/docs/#installation). Once you've done so, you can install the dependencies for the examples by running the following command in the current directory:

```bash
# verify poetry is installed
poetry --version

# install dependencies
poetry install

```

After installing the dependencies, you can run the examples by running the following command within the example directory you want to run:

```bash

# run the example
poetry run snakemake -j 1

# or to run the example on a slurm cluster
poetry run snakemake --executor slurm -j 1

```

## Cleaning up

To clean up the example directory, you can run the following command in the directory of the example you want to clean up:

```bash

# clean up the example
poetry run snakemake clean

```
