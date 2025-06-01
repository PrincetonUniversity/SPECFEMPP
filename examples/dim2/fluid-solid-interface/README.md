# Wave propagration through fluid-solid interface

This example creates the fluid-solid example with flat ocean bottom from
[Komatitsch et. al.](https://doi.org/10.1190/1.1444758). For a step-by-step
guide on this example, please refer to the
[documentation](https://specfem2d-kokkos.readthedocs.io/en/latest/cookbooks/dim2/fluid-solid-interface/index.html).

## Running the examples

Add path/to/SPECFEMPP/bin to your PATH environment variable, so that you can
run the `specfem2d` command from anywhere.

```bash
export PATH=$PATH:path/to/SPECFEMPP/bin
```

Create the output directory:

```bash
mkdir -p OUTPUT_FILES/results
```

Compute the mesh using the `xmeshfem2d` command:

```bash
xmeshfem2d -p Par_file
```

Run the simulation:

```bash
specfem2d -p specfem_config.yaml
```

The output seismograms will be stored in the `OUTPUT_FILES/results` directory.
Each station has a two column file with the first column being the time and the
second column being the amplitude of the wave at that time.

## Cleaning up

To clean up this example directory, you can run the following command in the
directory of the example you want to clean up:

```bash
rm -rf OUTPUT_FILES
```
