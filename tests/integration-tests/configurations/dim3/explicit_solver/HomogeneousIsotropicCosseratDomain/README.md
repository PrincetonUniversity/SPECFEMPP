Seismograms are computed from the analytic solution using [cosserat-solver](https://github.com/maxlchien/Cosserat-Solver) using the configurations in `examples/cosserat_3d_benchmark`, but with an `extension_factor` of 128 instead of 8. To reproduce the seismograms at `extension_factor=8`, run the following command:
```sh
git clone git@github.com:maxlchien/Cosserat-Solver.git
cd Cosserat-Solver
uv pip install -e .
cd examples/cosserat_3d_benchmark
uv run cosserat-solver --yaml params.yaml
```
To reproduce the seismograms at `extension_factor=128`, edit the `simulation_params` block in `params.yaml` to be
```yaml
simulation_params:
  dt: 0.01
  N: 5000
  refinement_factor: 1
  extension_factor: 128
```
To see accuracy metrics, run `snakemake -c1`.
