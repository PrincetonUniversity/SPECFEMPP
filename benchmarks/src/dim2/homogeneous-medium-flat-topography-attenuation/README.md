# Wave propagation through homogeneous medium with/without attenuation

This is the exact same mesh as in the homogeneous elastic medium with flat topography example,
but now we simulate wave propagation through the medium both with and without attenuation.

## Running the examples

To run the examples, you first need to install uv following these
[instructions](https://docs.astral.sh/uv/getting-started/installation). Once you've done
so, you can install the dependencies for the examples by running the following
command in the current directory:

```bash
# verify uv is installed
uv --version

# install dependencies
uv sync --group examples

```

After installing the dependencies, you can run the examples by running the following command within the example directory you want to run:

```bash

# run the example
uv run snakemake -j 2

# or to run the example on a slurm cluster
uv run snakemake --executor slurm -j 2

```

Set to `-j` to 2 if you want to run both the attenuation and non-attenuation cases in
parallel, otherwise set to 1.

## Cleaning up

To clean up the example directory, you can run the following command in the directory of the example you want to clean up:

```bash
# clean up the example
uv run snakemake clean -j 1
```
