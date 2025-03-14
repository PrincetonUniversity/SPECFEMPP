# Example of a homogeneous half-space in 3D

## Meshing

```bash
mkdir -p OUTPUT_FILES/DATABASES_MPI
xmeshfem3D -p DATA/meshfem3D_files/Mesh_Par_file
xgenerate_databases -p Par_File
```

## Running the simulation (so far only reading the mesh)
Then we can run specfem with the following command:

```bash
specfem3d -p specfem_config.yaml
```
