# CUBIT example for SPECFEMPP

This example demonstrates how to use the CUBIT mesh generator with SPECFEMPP. For a step-by-step guide on this example, please refer to the [documentation](https://specfem2d-kokkos.readthedocs.io/en/latest/cookbooks/dim2/CUBIT/index.html).


> **Note:** This example requires the CUBIT mesh generator to be installed on your system. If you do not have CUBIT installed, you can download it from [here](https://cubit.sandia.gov/).

> **Note:** This example requires that SPECFEM++ is installed with VTK support. Please see the [installation guide](https://specfem2d-kokkos.readthedocs.io/en/latest/sections/getting_started/index.html#optional)

## Running the examples

Add path/to/SPECFEMPP/bin to your PATH environment variable, so that you can
run the `specfem2d` command from anywhere.

```bash
export PATH=$PATH:path/to/SPECFEMPP/bin
```

Create the output directory:

```bash
mkdir -p OUTPUT_FILES/results
mkdir -p OUTPUT_FILES/display
```

Compute the mesh using the `xmeshfem2d` command:

```bash
xmeshfem2d -p Par_file
```

Run the simulation:

```bash
specfem2d -p specfem_config.yaml
```

The output wavefield will be stored in the `OUTPUT_FILES/display` directory.

## Cleaning up

To clean up this example directory, you can run the following command in the
directory of the example you want to clean up:

```bash
rm -rf OUTPUT_FILES
```
