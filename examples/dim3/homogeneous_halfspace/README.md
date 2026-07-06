# Wave propagation through a homogeneous elastic halfspace

In this example we simulate wave propagation through a 3-dimensional
homogeneous elastic halfspace using a vertical force source. The model has a
flat free surface, one elastic material, and no internal interfaces or
attenuation.

For a step-by-step guide on this example, please refer to the
[documentation][homogeneous-halfspace-docs].

[homogeneous-halfspace-docs]: https://specfem2d-kokkos.readthedocs.io/en/latest/cookbooks/dim3/homogeneous-isotropic-force/index.html

## Running the example

Add `path/to/SPECFEMPP/bin` to your `PATH` environment variable, so that you can
run the `xmeshfem3D` and `specfem` commands from anywhere.

```bash
export PATH=$PATH:path/to/SPECFEMPP/bin
```

Create the output directories:

```bash
mkdir -p OUTPUT_FILES/DATABASES
mkdir -p OUTPUT_FILES/results
```

Compute the mesh using the `xmeshfem3D` command:

```bash
xmeshfem3D -p DATA/meshfem3D_files/Mesh_Par_file
```

Run the simulation:

```bash
specfem 3d -p specfem_config.yaml
```

The output seismograms will be stored in the `OUTPUT_FILES/results` directory.
Each station has a two-column file with the first column being time and the
second column being displacement amplitude.

## Plotting

To plot the source-station geometry and generated seismograms, run:

```bash
python3 plot_seismograms.py
```

The plots will be written to `OUTPUT_FILES/geometry.png` and
`OUTPUT_FILES/seismograms.png`.

## Cleaning up

To clean up this example directory, run:

```bash
rm -rf OUTPUT_FILES
```
