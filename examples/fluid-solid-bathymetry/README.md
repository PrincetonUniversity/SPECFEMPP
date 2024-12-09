# Wave propagration through fluid-solid interface

This example simulates a tele-seismic plane wave within a fluid-solid domain. This example is contributed by Sirawich Pipatprathanporn and is part of the publication [Pipatprathanporn et al. (2024)](https://doi.org/10.1093/gji/ggae238)

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
