# Wave propagration through fluid-solid interface

This example simulates a tele-seismic plane wave within a fluid-solid domain.
This example is contributed by Sirawich Pipatprathanporn and is part of the
publication [Pipatprathanporn et al.
(2024)](https://doi.org/10.1093/gji/ggae238)

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


## Create animated gif

**note**: Requires `ImageMagick`

First create a directory to store the cropped images:

```bash
mkdir output_cropped
```

Then, crop the images to remove most of the white border (smaller disk

```bash
for file in $(ls OUTPUT_FILES/display/wavefield*.png);
  do bname=$(basename $file);
  magick $file -crop 2260x1110+150+725 output_cropped/$bname;
done;
```

Finally, create an animated gif:

```bash
magick convert -coalesce -delay 1 -loop 0 output_cropped/wavefield*.png fluid-solid-bathymetry.gif
```

## Cleaning up

To clean up the example directory, you can run the following command in the
directory of the example you want to clean up:

```bash

# clean up the example
uv run snakemake clean -j 1

```
