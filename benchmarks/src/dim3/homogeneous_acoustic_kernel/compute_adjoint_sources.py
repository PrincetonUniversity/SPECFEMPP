"""
compute_adjoint_sources.py

Compute cross-correlation traveltime adjoint sources from forward pressure
seismograms (Tromp, Tape & Liu, GJI 2005).

Usage:
    python compute_adjoint_sources.py <seismogram_dir> <output_dir> \
        <seismogram_basename> <t_window_start> <t_window_end>

Example:
    python compute_adjoint_sources.py OUTPUT_FILES/results adjoint_sources \
        DB.X20.S3.MXP.semp 14.0 24.0

The adjoint source for cross-correlation traveltime misfit is:

    f†(t) = -1/N * dp/dt(t) * w(t)

where:
    p(t)  = forward pressure seismogram
    w(t)  = Hanning taper window centred on [t_start, t_end]
    N     = ∫ [dp/dt(t)]^2 dt  (normalization)

Output file has the same time axis as the input seismogram (same t0, dt,
nstep) so that the External STF reader accepts it.
"""

import sys
import numpy as np
from pathlib import Path


def hanning_window(t, t_start, t_end):
    """Return a Hanning taper in [t_start, t_end], zero outside."""
    win = np.zeros_like(t)
    mask = (t >= t_start) & (t <= t_end)
    duration = t_end - t_start
    win[mask] = 0.5 * (1.0 - np.cos(2.0 * np.pi * (t[mask] - t_start) / duration))
    return win


def compute_adjoint_source(
    seismo_file: Path, output_file: Path, t_start: float, t_end: float
) -> None:
    """Compute adjoint source and write to file."""
    # Load seismogram: two columns (time, pressure)
    data = np.loadtxt(seismo_file)
    t = data[:, 0]
    p = data[:, 1]
    dt = t[1] - t[0]

    # Time-domain derivative of pressure (central differences)
    dpdt = np.gradient(p, dt)

    # Hanning taper window
    win = hanning_window(t, t_start, t_end)

    # Windowed velocity
    dpdt_win = dpdt * win

    # Normalization
    N = np.trapz(dpdt_win**2, dx=dt)
    if abs(N) < 1.0e-30:
        print(
            f"WARNING: normalization is near zero for {seismo_file.name}. "
            f"Ensure the time window [{t_start}, {t_end}] overlaps the arrival."
        )
        N = 1.0

    # Adjoint source
    adj = -dpdt_win / N

    # Write output: same time axis
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for ti, ai in zip(t, adj):
            f.write(f"{ti:.6e}  {ai:.6e}\n")

    print(
        f"Written adjoint source: {output_file}  (max|adj|={np.max(np.abs(adj)):.3e})"
    )


def main():
    if len(sys.argv) != 6:
        print(__doc__)
        sys.exit(1)

    seismo_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    seismo_basename = sys.argv[3]  # e.g. DB.X20.S3.MXP.semp
    t_start = float(sys.argv[4])
    t_end = float(sys.argv[5])

    seismo_file = seismo_dir / seismo_basename

    # Output file: same name but .adj extension
    stem = seismo_basename.rsplit(".", 1)[0]  # remove extension
    adj_file = output_dir / f"{stem}.adj"

    if not seismo_file.exists():
        print(f"ERROR: seismogram file not found: {seismo_file}")
        sys.exit(1)

    compute_adjoint_source(seismo_file, adj_file, t_start, t_end)


if __name__ == "__main__":
    main()
