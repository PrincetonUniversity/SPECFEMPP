

# Examples

The examples use [snakemake](https://snakemake.readthedocs.io/en/stable/) to run the simulation workflow. Snakemake is a workflow management system that allows you to define the rules of your workflow in a Snakefile. Every example has its own Snakefile that defines the steps involved in running that example.

**Make sure you've installed SPECFEM++ with examples enabled (-D ENABLE_EXAMPLES=ON)**

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

cd homogeneous-medium-flat-topography

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
