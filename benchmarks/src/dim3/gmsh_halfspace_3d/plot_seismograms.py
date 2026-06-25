import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
import os


def read_stations(filename):
    stations = []
    with open(filename, "r") as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                stations.append(
                    {
                        "station": parts[0],
                        "network": parts[1],
                        "x": float(parts[2]),
                        "y": float(parts[3]),
                        "elevation": float(parts[4]),
                        "burial": float(parts[5]),
                    }
                )
    return stations


def read_seismogram(filename):
    data = np.loadtxt(filename)
    return data[:, 0], data[:, 1]


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    stations = read_stations(os.path.join(base_dir, "DATA", "STATIONS"))

    # Source is at x=5000, y=5000 (from sources.yaml)
    source_x, source_y = 5000.0, 5000.0

    for station in stations:
        station["distance"] = np.sqrt(
            (station["x"] - source_x) ** 2 + (station["y"] - source_y) ** 2
        )

    stations_sorted = sorted(stations, key=lambda s: s["distance"])

    results_dir = os.path.join(base_dir, "OUTPUT_FILES", "results")

    # Band code C: dt=0.004 s -> fs=250 Hz
    components = ["CXX", "CXY", "CXZ"]

    seismogram_files = {
        c: sorted(glob.glob(os.path.join(results_dir, f"*.S3.{c}.semd")))
        for c in components
    }

    fig = plt.figure(figsize=(20, 6))
    gs = gridspec.GridSpec(1, 4, width_ratios=[2, 1, 1, 1], hspace=0.3, wspace=0.3)

    ax_map = fig.add_subplot(gs[0, 0])
    for station in stations:
        ax_map.plot(station["x"], station["y"], "rv", markersize=8)
        ax_map.text(
            station["x"],
            station["y"] + 200,
            station["station"],
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax_map.plot(source_x, source_y, "r*", markersize=15, label="Source")
    ax_map.set_xlabel("X (m)")
    ax_map.set_ylabel("Y (m)")
    ax_map.set_title("Source-Station Geometry")
    ax_map.legend()
    ax_map.grid(True, alpha=0.3)
    ax_map.set_aspect("equal")

    all_seismograms = {}
    time_range = None
    max_displacement = 0.0

    for component in components:
        all_seismograms[component] = {}
        for filename in seismogram_files[component]:
            station_name = os.path.basename(filename).split(".")[1]
            time, displacement = read_seismogram(filename)
            all_seismograms[component][station_name] = (time, displacement)
            if time_range is None:
                time_range = (time.min(), time.max())
            else:
                time_range = (
                    min(time_range[0], time.min()),
                    max(time_range[1], time.max()),
                )
            max_displacement = max(max_displacement, np.abs(displacement).max())

    if max_displacement == 0.0:
        max_displacement = 1.0

    for i, component in enumerate(components):
        ax = fig.add_subplot(gs[0, i + 1])
        y_spacing = max_displacement * 2.5

        for j, station in enumerate(stations_sorted):
            station_name = station["station"]
            if station_name in all_seismograms[component]:
                time, displacement = all_seismograms[component][station_name]
                normalized = displacement / max_displacement * y_spacing * 0.8
                ax.plot(time, normalized + j * y_spacing, "k-", linewidth=0.8)
            ax.text(
                time_range[0] if time_range else 0,
                j * y_spacing,
                f"{station_name}\n({station['distance'] / 1000:.1f} km)",
                ha="right",
                va="center",
                fontsize=7,
            )

        ax.set_xlabel("Time (s)")
        ax.set_title(f"Component {component}")
        if time_range:
            ax.set_xlim(time_range)
            ax.set_ylim(
                -y_spacing * 0.5,
                (len(stations_sorted) - 1) * y_spacing + y_spacing * 0.5,
            )
        ax.grid(True, alpha=0.3)
        ax.set_yticklabels([])

    plt.tight_layout()
    out_path = os.path.join(base_dir, "OUTPUT_FILES", "seismogram_plot.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.show(block=False)


if __name__ == "__main__":
    main()
