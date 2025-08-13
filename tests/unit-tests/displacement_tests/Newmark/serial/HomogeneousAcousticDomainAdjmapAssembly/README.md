# Testing with the adjacency map

The only difference between this test and `HomogeneousAcousticDomain` is the existence (and setting true) of the `write_adjacency_map` flag.

```parfile
write_adjacency_map = .true.
```

This tells `meshfem` to export the adjacency map into the database, which flags `specfem` to construct the assembly from the adjacency map.

To reproduce the database, use the `HomogeneousAcousticDomain` provenance files, but include `write_adjacency_map = .true.` into `Par_file`. Sources and traces already sample from the `HomogeneousAcousticDomain` directory, so they do not need to be recreated.

At present, we do not perform a check for whether the adjacency map → assembly index mapping routine is used -- the only check is if this database produces the same traces.
