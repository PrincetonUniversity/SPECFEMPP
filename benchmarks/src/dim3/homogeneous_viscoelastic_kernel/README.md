# Wave propagation through homogeneous, viscoelastic halfspace medium

This is a 3D homogeneous halfspace simulation with attenuation enabled. We do _not_ have:

- any topography (the surface is flat)
- any internal interfaces (the medium is homogeneous)

The material uses low quality factors so attenuation has a visible impact on the resulting kernel plots.

The adjoint configuration can subdivide each undo-attenuation replay buffer
under `solver.time-marching.checkpointing`:

```yaml
checkpointing:
  subdivide-buffer: 4  # four 64-step leaves for a 256-step window
```

The default, `subdivide-buffer: 1`, buffers every displacement in a disk
checkpoint window. Values greater than one retain the forward state between
smaller displacement buffers. Leaf size is the checkpoint-window length
divided by `subdivide-buffer`, rounded up when it is not exact. The solver
derives one full-state checkpoint for each internal leaf boundary.

## Running the examples

To run any example, you first need to install uv following these
[instructions](https://docs.astral.sh/uv/getting-started/installation). Once you've done
so, you can install the dependencies for the examples by running the following
command in the current directory:

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

To clean up the example directory, you can run the following command in the directory of the example you want to clean up:

```bash

# clean up the example
uv run snakemake clean -j 1

```
