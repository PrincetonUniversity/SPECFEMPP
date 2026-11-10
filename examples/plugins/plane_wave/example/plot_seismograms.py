import glob
import os

import matplotlib.pyplot as plt
import numpy as np


def read_stations(filename):
    """Read the STATIONS file (name, network, y, x, elevation, burial)."""
    stations = []
    with open(filename) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            stations.append(
                {
                    "name": parts[0],
                    "network": parts[1],
                    "y": float(parts[2]),
                    "x": float(parts[3]),
                }
            )
    return stations


def read_seismogram(network, station, component):
    pattern = f"OUTPUT_FILES/results/{network}.{station}.*.BX{component}.semd"
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No seismogram matches {pattern}. Did the simulation run?"
        )
    data = np.loadtxt(files[0])
    return data[:, 0], data[:, 1]


def main():
    stations = read_stations("DATA/STATIONS")

    # Order the stations along the line they sit on.
    x0, y0 = stations[0]["x"], stations[0]["y"]
    for station in stations:
        station["offset"] = np.hypot(station["x"] - x0, station["y"] - y0)
    stations.sort(key=lambda s: s["offset"])

    components = ["X", "Y", "Z"]
    seismograms = {
        (s["name"], c): read_seismogram(s["network"], s["name"], c)
        for s in stations
        for c in components
    }
    peak = max(np.abs(u).max() for _, u in seismograms.values())
    spacing = 1.2 * peak

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    for ax, component in zip(axes, components):
        for i, station in enumerate(stations):
            time, displacement = seismograms[(station["name"], component)]
            ax.plot(time, displacement + i * spacing, "k-", linewidth=0.8)
            ax.text(
                time[0],
                i * spacing,
                f"{station['name']} ",
                ha="right",
                va="center",
                fontsize=8,
            )
        ax.set_xlabel("Time (s)")
        ax.set_title(f"Displacement, component {component}")
        ax.grid(True, alpha=0.3)
        ax.set_yticks([])

    fig.suptitle(
        "Plane-wave injection: stations ordered along the surface line "
        "(offset from the first station increases upwards)"
    )
    os.makedirs("OUTPUT_FILES", exist_ok=True)
    plt.savefig("OUTPUT_FILES/seismograms.png", dpi=200, bbox_inches="tight")
    print("Saved seismograms plot to OUTPUT_FILES/seismograms.png")


if __name__ == "__main__":
    main()
