# Mesh for wavepropagation in a 2D homogeneous Cosserat medium

This mesh is used for testing the assembly of the spectral element method mesh.
This mesh was generated assuming that cosserat medium parameters are correctly
read and defined in the SPECFEM++ version of the mesher.

# Provenance

This mesh was generated using the SPECFEM++ mesher with the following command:

```bash
cd provenance
mkdir -p OUTPUT_FILES/results
xmeshfem2d -p Par_file
```

This should generate both the `database.bin` file and the `STATIONS` file.

The simulation should then be run with the following command:

```bash
cd ..
specfem2d -p specfem_config.yaml
```

in the figure below, we show the source-station geometry.

![Source-Station Geometry](provenance/geometry.png)

The generated seismograms for the source given in the `source.yaml` file
are shown in the figure below.

![Seismograms](provenance/seismograms.png)
