# Anisotropic Crystal example

This example demonstrates how to run wave prpagation simulations in an
anisotropic crystal.

```bash
# activate the conda environment with the required dependencies.
cd path/to/examples/anisotropic-crystal
```

Then, run the example

```bash
snakemake
```

This will run the mesher, then the simulation and generate the output files in
the `OUTPUT_FILES/results` directory. The output files are:

- `plot.png`: a plot of 5 selected traces.
- `AA.S00{01..50}.S2.BX{X,Z}.semd`: displacement seismograms for the 50 stations.

## Auxiliary plotting

Some plotting for presentations and reports can be done with the following
additions.

### Source-Station geometry

To plot the station for this example, please use

```bash
python extras/plot_geometry.py
```

which produces `geometry.png`.


### The wavefield

If you want to plot the wavefield during the simulation you can edit the
`specfem_config.yaml` in the example directory, and add the following lines
to the writer section:

```yaml
    # ...
    simulation-mode:
      forward:
        writer:
          # ...  other writers

          # visualization writer
          display:
              format: PNG
              directory: "/Users/lsawade/SPECFEMPP/examples/anisotropic-crystal/OUTPUT_FILES/results"
              field: displacement
              simulation-field: forward
              time-interval: 25
    # ...
```

Then, run `snakemake clean` and `snakemake` again. The wavefield will be
saved in the form of snapshots the `OUTPUT_FILES/results` directory.

If you have `ImageMagick` installed, you can first crop the images to remove
most of the white border (smaller disk footprint) with the following command:

```bash
mkdir output_cropped
for file in $(ls OUTPUT_FILES/results/wavefield*.png);
  do bname=$(basename $file);
  magick $file -crop 1760x1760+400+400 output_cropped/$bname;
done;
```

And then, use the following command to create an animated gif:

```bash
convert -coalesce -delay 1 -loop 0 output_cropped/wavefield*.png anisotropic-crystal.gif
```

This will create a file `anisotropic-crystal.gif` that shows the wavefield
propagation during the simulation.
