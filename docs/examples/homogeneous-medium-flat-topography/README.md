# Wave propagation through homogeneous medium with no interfaces

In this example we simulate wave propagation through a 2-dimensional homogeneous medium.

## Generating the mesh

To generate the mesh for the homogeneous media we need a parameter file, `Par_File`, a topography file, `topography_file.dat`, and the mesher executible, `xmeshfem2D`, which should have been compiled during the installation process.

>  Currently, we still use a mesher that was developed for the original [SPECFEM2D](https://specfem2d.readthedocs.io/en/latest/03_mesh_generation/) code. More details on the meshing process can be found [here](https://specfem2d.readthedocs.io/en/latest/03_mesh_generation/).

## Running the mesher

To execute the mesher run

```
    ./xmeshfem2D -p <PATH TO PAR_FILE>
```

> Make sure either your are in the build directory of SPECFEM2D kokkos or the build directory is added to your ``PATH``.

Note the path of the database file and :ref:`stations_file` generated after successfully running the mesher.

## Running the solver

Finally, to run the SPECFEM2D kokkos solver

```
    ./specfem2d -p <PATH TO specfem_config.yaml>
```
