"""Capture independent mesher reference values when regenerating the fixture."""

import re
import sys
from pathlib import Path

log = Path(sys.argv[1]).read_text()
volume = re.search(r"calculated volume:\s+(\S+)", log).group(1)
mass = re.search(r"computed total Earth mass[^\n]*\n\s*(\S+)\s+kg", log).group(1)
dt = re.search(r"DT =\s+(\S+)", log).group(1)
count = re.search(r"total number of elements in all slices:\s+(\d+)", log).group(1)
Path(sys.argv[2]).write_text(
    "# Extracted from finalize_mesher in OUTPUT_FILES/output_mesher.txt.\n"
    "# The reported volume is nondimensional; multiply by r_planet**3.\n"
    f"volume_unit_sphere: {volume}\nmass_kg: {mass}\n"
    f"dt_seconds: {dt}\nmesher_element_count: {count}\n"
    "# PREM_RCMB and PREM_RICB from shared/model_prem.f90.\n"
    "r_cmb_m: 3480000.0\nr_icb_m: 1221500.0\n"
)
report = log[log.index(" calculated volume:") : log.index(" End of mesh generation")]
Path(sys.argv[3]).write_text(
    "\n".join(line.rstrip() for line in report.splitlines()).rstrip() + "\n"
)
