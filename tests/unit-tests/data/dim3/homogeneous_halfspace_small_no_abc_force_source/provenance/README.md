# Homogeneous halfspace with force source, small model, no ABC

```bash
mkdir -p OUTPUT_FILES/DATABASES_MPI
```

Here, you'll have to create the `bin` folder and link the executables from your `specfem3d` compilation.

```bash
mkdir -p bin
cd bin
ln -s /path/to/your/compiled/specfem3d/bin/xmeshfem3D
ln -s /path/to/your/compiled/specfem3d/bin/xgenerate_databases
ln -s /path/to/your/compiled/specfem3d/bin/xspecfem3D
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
