# Wave propagation through a homogeneous Cosserat medium

3D wave propagation in a homogeneous, isotropic elastic Cosserat (micropolar)
medium. We do _not_ have:

- any topography (the surface is flat)
- any internal interfaces (the medium is homogeneous)
- absorbing boundaries (free surfaces on all sides)

A Cosserat force + couple source is buried at the center of a 50 km cube, and
buried receivers around it record displacement, rotation, intrinsic rotation,
and curl seismograms. The material, source, and station parameters follow the
analytic validation setup of the `HomogeneousIsotropicCosseratDomain`
integration test (issue #1815), which benchmarks against Green's functions
from Eringen (1999).

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

## Wavefield snapshots (optional)

The solver can write rotation wavefield snapshots to a VTKHDF file that can be
opened in ParaView. This requires SPECFEM++ to be built with HDF5 support. To
enable it, uncomment the `display:` block in `specfem_config.yaml` and the
`wavefield.vtkhdf` lines in the `Snakefile` (in the benchmark build directory),
then rerun the workflow. The `field:` and `component:` keys select what is
plotted, e.g. `rotation` / `magnitude`.

## Cleaning up

To clean up the example directory, you can run the following command in the directory of the example you want to clean up:

```bash

# clean up the example
uv run snakemake clean -j 1

```
