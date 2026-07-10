"""
Compute cross-correlation traveltime adjoint sources for elastic kernels.

Usage:
    python compute_adjoint_sources.py <results_dir> <adjoint_dir> <trace_filename>

The adjoint source for traveltime misfit (Tromp et al. 2005) is:
    f†(t) = -du/dt / ∫(du/dt)² dt
where u(t) is one displacement component.
"""

import sys
import os
import numpy as np


def compute_adjoint_source(time, displacement):
    dt = time[1] - time[0]
    du_dt = np.gradient(displacement, dt)
    norm = np.trapezoid(du_dt**2, time)
    if norm == 0.0:
        return np.zeros_like(displacement)
    return -du_dt / norm


def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <results_dir> <adjoint_dir> <trace_filename>")
        sys.exit(1)

    results_dir = sys.argv[1]
    adjoint_dir = sys.argv[2]
    trace_filename = sys.argv[3]

    trace_path = os.path.join(results_dir, trace_filename)
    data = np.loadtxt(trace_path)
    time, displacement = data[:, 0], data[:, 1]

    adj = compute_adjoint_source(time, displacement)

    adj_filename = trace_filename.replace(".semd", ".adj")
    adj_path = os.path.join(adjoint_dir, adj_filename)
    np.savetxt(adj_path, np.column_stack([time, adj]))
    print(f"Written: {adj_path}")


if __name__ == "__main__":
    main()
