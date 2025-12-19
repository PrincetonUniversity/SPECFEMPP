# Homogeneous halfspace with force source, small model, no ABC

## Recreating the traces

Use `provenance/fortran`.

```bash
mkdir -p OUTPUT_FILES/DATABASES_MPI
```

Here, you'll have to create the `bin` folder and link the executables from your `specfem3d` compilation.

```bash
mkdir -p bin
cd bin
SPECFEM3D="/path/to/your/compiled/specfem3d"
ln -s ${SPECFEM3D}/bin/xmeshfem3D
ln -s ${SPECFEM3D}/bin/xgenerate_databases
ln -s ${SPECFEM3D}/bin/xspecfem3D
cd -
```

Then, you can run the simulation with the following commands:

```bash
mpirun -np 1 ./bin/xmeshfem3D
```

```bash
mpirun -np 1 ./bin/xgenerate_databases
```

```bash
mpirun -np 1 ./bin/xspecfem3D
```

Then move the traces from `OUTPUT_FILES` to the `traces` directory.

```bash
mv -f OUTPUT_FILES/*.sem* ../../traces/
```

## Recreating The SPECFEM++ databases.

Use `provenance/specfempp`

Since the database is written slightly different from the fortran version you
will have to recreate the databases with the SPECFEM++ mesher.

Add specfemmpp to your path
```bash
export PATH=path/to/specfempp/bin/:${PATH}
```
```bash
cd provenance/specfempp
mkdir -p OUTPUT_FILES/DATABASES_MPI
xmeshfem3D -p DATA/meshfem3D_files/Mesh_Par_file
```

Which should create `OUTPUT_FILES/DATABASES_MPI/proc000000_Database`.

Finally, move `proc000000_Database` to the test directory.

```bash
cp OUTPUT_FILES/DATABASES_MPI/proc000000_Database ../../mesh_database.bin
```
