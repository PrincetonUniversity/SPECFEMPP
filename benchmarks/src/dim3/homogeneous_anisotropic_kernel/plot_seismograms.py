import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
import yaml


def read_stations(filename):
    """Read STATIONS file"""
    stations = []
    with open(filename, "r") as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                station = parts[0]
                network = parts[1]
                y = float(parts[2])  # latitude/UTM_Y
                x = float(parts[3])  # longitude/UTM_X
                elevation = float(parts[4])
                burial = float(parts[5])
                stations.append(
                    {
                        "station": station,
                        "network": network,
                        "x": x,
                        "y": y,
                        "elevation": elevation,
                        "burial": burial,
                    }
                )
    return stations


def read_force_yaml(filename):
    """Read source location from force.yaml"""
    with open(filename, "r") as f:
        data = yaml.safe_load(f)
    src = data["sources"][0]["force"]
    return {"x": src["x"], "y": src["y"], "z": src["z"]}


def read_seismogram(filename):
    """Read seismogram file"""
    data = np.loadtxt(filename)
    return data[:, 0], data[:, 1]  # time, displacement


def calculate_epicentral_distance(station_x, station_y, source_x, source_y):
    """Calculate epicentral distance"""
    return np.sqrt((station_x - source_x) ** 2 + (station_y - source_y) ** 2)


def main():
    # Read station and source data
    stations = read_stations("DATA/STATIONS")
    source = read_force_yaml("forward_source.yaml")

    # Calculate epicentral distances and sort stations
    for station in stations:
        station["distance"] = calculate_epicentral_distance(
            station["x"], station["y"], source["x"], source["y"]
        )

    stations_sorted = sorted(stations, key=lambda x: x["distance"])

    # Create figure with gridspec for 1 row x 2 columns
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], hspace=0.3, wspace=0.4)

    # Subplot 1: Source-Station geometry
    ax1 = fig.add_subplot(gs[0, 0])

    for station in stations:
        ax1.plot(
            station["x"],
            station["y"],
            "rv",
            markersize=8,
            label="Stations" if station == stations[0] else "",
        )
        ax1.text(
            station["x"],
            station["y"] + 1000,
            station["station"],
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax1.plot(source["x"], source["y"], "r*", markersize=15, label="Source")

    ax1.set_xlim(0, 150000)
    ax1.set_ylim(0, 100000)
    ax1.set_xticks(range(0, 150001, 20000))
    ax1.set_xticklabels([f"{x // 1000}km" for x in range(0, 150001, 20000)])
    ax1.set_yticks(range(0, 100001, 20000))
    ax1.set_yticklabels([f"{y // 1000}km" for y in range(0, 100001, 20000)])
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_title("Source-Station Geometry")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Read BXX displacement seismograms
    seismogram_files = sorted(glob.glob("OUTPUT_FILES/results/*.S3.BXX.semd"))
    reference_files = sorted(glob.glob("reference_seismograms/*.S3.BXX.semd"))

    all_seismograms = {}
    reference_seismograms = {}
    time_range = (0.0, 1.0)
    max_displacement = 0

    for filename in seismogram_files:
        station_name = filename.split("/")[-1].split(".")[1]
        time, displacement = read_seismogram(filename)
        all_seismograms[station_name] = (time, displacement)
        time_range = (min(time_range[0], time.min()), max(time_range[1], time.max()))
        max_displacement = max(max_displacement, np.abs(displacement).max())

    for filename in reference_files:
        station_name = filename.split("/")[-1].split(".")[1]
        time, displacement = read_seismogram(filename)
        reference_seismograms[station_name] = (time, displacement)
        time_range = (min(time_range[0], time.min()), max(time_range[1], time.max()))
        max_displacement = max(max_displacement, np.abs(displacement).max())

    # Subplot 2: Displacement seismograms
    ax2 = fig.add_subplot(gs[0, 1])
    y_spacing = max_displacement * 2.5 if max_displacement > 0 else 1.0

    for j, station in enumerate(stations_sorted):
        station_name = station["station"]
        y_pos = j * y_spacing

        if station_name in all_seismograms:
            time, displacement = all_seismograms[station_name]
            normalized = (
                displacement / max_displacement * y_spacing * 0.8
                if max_displacement > 0
                else displacement
            )
            ax2.plot(
                time,
                normalized + y_pos,
                "k-",
                linewidth=0.8,
                label="specfem++" if j == 0 else "",
            )

        if station_name in reference_seismograms:
            time_ref, displacement_ref = reference_seismograms[station_name]
            normalized_ref = (
                displacement_ref / max_displacement * y_spacing * 0.8
                if max_displacement > 0
                else displacement_ref
            )
            ax2.plot(
                time_ref,
                normalized_ref + y_pos,
                "r--",
                linewidth=0.8,
                alpha=0.7,
                label="xspecfem3D" if j == 0 else "",
            )

        if station_name in all_seismograms or station_name in reference_seismograms:
            ax2.text(
                time_range[0] - (time_range[1] - time_range[0]) * 0.05,
                y_pos,
                f"{station_name}\n({station['distance'] / 1000:.1f}km)",
                ha="right",
                va="center",
                fontsize=8,
            )

    ax2.set_xlabel("Time (s)")
    ax2.set_title("Component BXX (Displacement)")
    ax2.set_xlim(time_range)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", fontsize=8, fancybox=False)
    if len(stations_sorted) > 0:
        ax2.set_ylim(
            -y_spacing * 0.5,
            (len(stations_sorted) - 1) * y_spacing + y_spacing * 0.5,
        )
    ax2.set_yticklabels([])
    ax2.set_aspect("auto")

    plt.tight_layout()
    plt.savefig("OUTPUT_FILES/seismogram_plot.png", dpi=300, bbox_inches="tight")
    plt.show(block=False)


if __name__ == "__main__":
    main()
