# Acoustic-elastic force source

This example simulates wave propagation across a horizontal acoustic-elastic
interface in 3D. The lower layer is elastic, the upper layer is acoustic, and a
vertical force source is placed in the elastic domain.

This example is adapted from the `AcousticElasticForce` displacement test, with
a larger model domain, a longer simulation duration, and Stacey absorbing
boundaries enabled on the side and bottom boundaries.

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
The example writes displacement, velocity, acceleration, and pressure traces.

## Plotting

To plot the source-station geometry and generated seismograms, run:

```bash
python3 plot_seismograms.py
```

The plots will be written to `OUTPUT_FILES/geometry.png`,
`OUTPUT_FILES/displacement_seismograms.png`,
`OUTPUT_FILES/velocity_seismograms.png`,
`OUTPUT_FILES/acceleration_seismograms.png`, and
`OUTPUT_FILES/pressure_seismograms.png`.

## Model

The model spans `400 km x 300 km x 140 km`. The acoustic-elastic interface is at
`z = -30000 m`, with the elastic layer below and the acoustic layer above.

Stacey absorbing boundary conditions are enabled in
`DATA/meshfem3D_files/Mesh_Par_file`. The top boundary remains a free surface.

## Cleaning up

To clean up this example directory, run:

```bash
rm -rf OUTPUT_FILES
```
