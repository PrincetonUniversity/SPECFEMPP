

# Examples

The examples use [snakemake](https://snakemake.readthedocs.io/en/stable/) to run the simulation workflow. Snakemake is a workflow management system that allows you to define the rules of your workflow in a Snakefile. Every example has its own Snakefile that defines the steps involved in running that example.

**Make sure you've installed SPECFEM++ with examples enabled (-D ENABLE_EXAMPLES=ON) and add the build directory to your `PATH`**

## Running the examples

To run the examples, you first need to install uv following these [instructions](https://docs.astral.sh/uv/getting-started/installation). Once you've done so, you can install the dependencies for the examples by running the following command in the current directory:

```bash
# verify uv is installed
uv --version

# install dependencies
uv sync --extra examples

```

After installing the dependencies, you can run the examples by running the following command within the example directory you want to run:

```bash

cd <example directory>

# run the example
uv run snakemake -j 1

# or to run the example on a slurm cluster
uv run snakemake --executor slurm -j 1

```

## Cleaning up

To clean up the example directory, you can run the following command in the directory of the example you want to clean up:

```bash

# clean up the example
uv run snakemake clean -j 1

```
