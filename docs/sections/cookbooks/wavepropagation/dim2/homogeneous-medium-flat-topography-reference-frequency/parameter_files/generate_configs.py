"""Generate one specfem_config.yaml per reference-frequency variant."""

import os
import numpy as np

F0 = 30.0  # Hz — center of the sweep
N_FREQS = 9  # 4 below + center + 4 above

freqs = np.logspace(np.log10(F0 / 10), np.log10(F0 * 10), N_FREQS)

with open("specfem_config.yaml.in") as fh:
    template = fh.read()

for i, freq in enumerate(freqs):
    tag = f"{i:02d}"
    outdir = f"OUTPUT_FILES/freq_{tag}/results"
    os.makedirs(outdir, exist_ok=True)
    config = template.format(reference_frequency=f"{freq:.6g}", freq_tag=tag)
    outfile = f"OUTPUT_FILES/freq_{tag}/specfem_config.yaml"
    with open(outfile, "w") as fh:
        fh.write(config)
    print(f"  {outfile}  (f_ref = {freq:.3g} Hz)")
