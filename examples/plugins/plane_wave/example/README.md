# Plane wave in a homogeneous halfspace

This example runs SPECFEM++ with the plane-wave injection plugin from start to
finish: build the plugin, mesh a small box, run the simulation, and plot the
seismograms. The whole run takes a few seconds on a laptop.

A P wave (take-off angle 20°, azimuth 30°, Ricker waveform with f0 = 0.5 Hz)
enters a 12 × 12 × 8.4 km homogeneous elastic box through its sides and
bottom. The top of the box is a free surface. Seven stations on the free
surface lie on a line along the wave's horizontal direction.

There is no source in the run. The plugin supplies the whole incident wave.

## Files

| File | Contents |
|---|---|
| `specfem_config.yaml` | The SPECFEM++ run configuration: time stepping, stations, and the `injection` section that selects how the incident field is applied |
| `plane_wave.yaml` | The plugin's own configuration: material, wave type and direction, waveform. See the [plugin README](../README.md) for every key |
| `DATA/meshfem3D_files/` | Mesher input: `Mesh_Par_file` and the interface files |
| `DATA/STATIONS` | The seven surface stations |
| `plot_seismograms.py` | Plots the seismograms |

Two files configure two different programs. `specfem_config.yaml` is read by
`specfem 3d` and `plane_wave.yaml` is read by the plugin, which `specfem 3d`
loads and hands the file to through `--injection-config`.

The material in `plane_wave.yaml` (`rho`, `vp`, `vs`) must equal the material
in `Mesh_Par_file`. The plugin compares them at setup and stops, naming the
first mismatched point, if they differ.

## Running the example

All commands below run from this directory (`examples/plugins/plane_wave/example`).

### 1. Build the plugin

The plugin is built against an installed copy of the plugin SDK. From the
SPECFEM++ repository root:

```bash
cmake --preset release
cmake --install build/release --component plugin_sdk --prefix <sdk>
cmake -S examples/plugins/plane_wave -B build/plane_wave -DCMAKE_PREFIX_PATH=<sdk>
cmake --build build/plane_wave
```

Replace `<sdk>` with any directory you can write to. The plugin library is
`build/plane_wave/libplane_wave.so`. Point a variable at it:

```bash
export PLUGIN=<repository root>/build/plane_wave/libplane_wave.so
```

### 2. Mesh the box

Add `<repository root>/bin` to your `PATH` so that `xmeshfem3D` and `specfem`
can be found, then create the output directories and run the mesher:

```bash
export PATH=$PATH:<repository root>/bin
mkdir -p OUTPUT_FILES/DATABASES OUTPUT_FILES/results
xmeshfem3D -p DATA/meshfem3D_files/Mesh_Par_file
```

`Mesh_Par_file` sets `STACEY_AS_INJECTION_BOUNDARY = .true.`, which makes every
Stacey face of the mesh an injection face. The plugin is asked for the wave on
those faces only. The mesh has 10 × 10 × 7 elements of about 1.2 km, which
resolves the shortest wavelength of the Ricker waveform.

### 3. Run the simulation

```bash
specfem 3d -p specfem_config.yaml \
  --injection-plugin $PLUGIN \
  --injection-config plane_wave.yaml
```

The log shows the plugin it loaded and the injection method:

```
Injection plugin
  Name         : plane-wave
  Version      : 1.0.0
Injection method
  Method        : stacey
  Points        : ...
```

The seismograms are written to `OUTPUT_FILES/results`, one file per station and
component. Each has two columns: time, and displacement.

### 4. Plot the seismograms

```bash
python3 plot_seismograms.py
```

This needs `numpy` and `matplotlib`. The figure is written to
`OUTPUT_FILES/seismograms.png`.

## What to expect

- The pulse reaches the first station at about 4.9 s and the last at about
  6.0 s. The delay across the line is the horizontal travel time of the wave.
- On the vertical component the amplitude is about twice the incident
  amplitude of the wave's vertical part (1e-3 × cos 20°). A wave arriving at a
  free surface doubles there.
- The signal after the main pulse is the wave reflected from the free surface.
  This mesh is coarse, so the reflection is not accurate. This example shows
  how to run a plugin. It does not check the accuracy of the solution.

## Changing the wave

Edit `plane_wave.yaml`. To switch to an SV wave, set `wave.type: SV`. To change
the direction, set `wave.take-off` and `wave.azimuth`. The plugin reads the
file at setup, so there is nothing to rebuild.

If you change the material, change `Mesh_Par_file` and `plane_wave.yaml` to
match, and mesh again. If you change `f0`, the mesh must still resolve the
shortest wavelength, about `vs / (2.5 · f0)`.

## Cleaning up

```bash
rm -rf OUTPUT_FILES
```
