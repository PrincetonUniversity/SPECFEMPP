import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
import yaml
from matplotlib.patches import Circle


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


def read_yaml_source(filename):
    """Read source parameters from YAML file"""
    with open(filename, "r") as f:
        data = yaml.safe_load(f)
    source_params = data["sources"][0]["force"]
    source = {
        "x": source_params["x"],
        "y": source_params["y"],
        "z": source_params["z"],
    }
    return source


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
    source = read_yaml_source("force.yaml")

    # Calculate epicentral distances and sort stations
    for station in stations:
        station["distance"] = calculate_epicentral_distance(
            station["x"], station["y"], source["x"], source["y"]
        )

    stations_sorted = sorted(stations, key=lambda x: x["distance"])

    # ========================================
    # Figure 1: Source-Station Geometry
    # ========================================
    fig1 = plt.figure(figsize=(8, 8))
    ax1 = fig1.add_subplot(111)

    # Plot stations
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

    # Plot source
    ax1.plot(source["x"], source["y"], "r*", markersize=15, label="Source")

    # Add circular distance grid
    max_dist = max([s["distance"] for s in stations])
    circles = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000]
    for radius in circles:
        if radius <= max_dist * 1.2:
            circle = Circle(
                (source["x"], source["y"]),
                radius,
                fill=False,
                linestyle="--",
                alpha=0.3,
                color="gray",
            )
            ax1.add_patch(circle)
            # Add distance labels
            ax1.text(
                source["x"] + radius * 0.7,
                source["y"] + radius * 0.7,
                f"{radius / 1000:.0f}km",
                fontsize=8,
                alpha=0.7,
            )

    ax1.set_xlabel("X (UTM)")
    ax1.set_ylabel("Y (UTM)")
    ax1.set_title("Source-Station Geometry")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal")

    plt.savefig("OUTPUT_FILES/geometry.png", dpi=300, bbox_inches="tight")
    print("Saved source-station geometry plot to OUTPUT_FILES/geometry.png")

    # Find seismogram files (current output)
    seismogram_files = {
        "MXX": sorted(glob.glob("OUTPUT_FILES/results/*.S3.MXX.semd")),
        "MXY": sorted(glob.glob("OUTPUT_FILES/results/*.S3.MXY.semd")),
        "MXZ": sorted(glob.glob("OUTPUT_FILES/results/*.S3.MXZ.semd")),
    }

    # Find reference seismogram files
    reference_seismogram_files = {
        "MXX": sorted(glob.glob("reference_seismograms/*.S3.MXX.semd")),
        "MXY": sorted(glob.glob("reference_seismograms/*.S3.MXY.semd")),
        "MXZ": sorted(glob.glob("reference_seismograms/*.S3.MXZ.semd")),
    }

    # Read all seismograms and find common time range
    all_seismograms = {}
    reference_seismograms = {}
    time_range = None
    max_displacement = 0

    for component in ["MXX", "MXY", "MXZ"]:
        all_seismograms[component] = {}
        reference_seismograms[component] = {}

        # Read current output seismograms
        for filename in seismogram_files[component]:
            # Extract station name from filename
            station_name = filename.split("/")[-1].split(".")[1]
            time, displacement = read_seismogram(filename)
            all_seismograms[component][station_name] = (time, displacement)

            # Update global ranges
            if time_range is None:
                time_range = (time.min(), time.max())
            else:
                time_range = (
                    min(time_range[0], time.min()),
                    max(time_range[1], time.max()),
                )
            max_displacement = max(max_displacement, np.abs(displacement).max())

        # Read reference seismograms
        for filename in reference_seismogram_files[component]:
            # Extract station name from filename
            station_name = filename.split("/")[-1].split(".")[1]
            time, displacement = read_seismogram(filename)
            reference_seismograms[component][station_name] = (time, displacement)

            # Update global ranges
            if time_range is None:
                time_range = (time.min(), time.max())
            else:
                time_range = (
                    min(time_range[0], time.min()),
                    max(time_range[1], time.max()),
                )
            max_displacement = max(max_displacement, np.abs(displacement).max())

    # ========================================
    # Figure 2: Seismograms
    # ========================================
    fig2 = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(1, 3, hspace=0.3, wspace=0.3)

    # Plot seismograms for each component
    components = ["MXX", "MXY", "MXZ"]

    for i, component in enumerate(components):
        ax = fig2.add_subplot(gs[0, i])

        # Plot seismograms sorted by epicentral distance
        y_spacing = max_displacement * 2.5

        for j, station in enumerate(stations_sorted):
            station_name = station["station"]

            # Plot current output seismogram
            if station_name in all_seismograms[component]:
                time, displacement = all_seismograms[component][station_name]

                # Normalize and offset displacement
                normalized_disp = displacement / max_displacement * y_spacing * 0.8
                y_pos = j * y_spacing

                ax.plot(
                    time,
                    normalized_disp + y_pos,
                    "k-",
                    linewidth=0.8,
                    label="specfem++" if j == 0 else "",
                )

            # Plot reference seismogram
            if station_name in reference_seismograms[component]:
                time_ref, displacement_ref = reference_seismograms[component][
                    station_name
                ]

                # Normalize and offset displacement
                normalized_disp_ref = (
                    displacement_ref / max_displacement * y_spacing * 0.8
                )
                y_pos = j * y_spacing

                ax.plot(
                    time_ref,
                    normalized_disp_ref + y_pos,
                    "r--",
                    linewidth=0.8,
                    alpha=0.7,
                    label="xspecfem3D" if j == 0 else "",
                )

            # Add station label and distance
            if (
                station_name in all_seismograms[component]
                or station_name in reference_seismograms[component]
            ):
                ax.text(
                    time_range[0] - (time_range[1] - time_range[0]) * 0.05,
                    y_pos,
                    f"{station_name}\n({station['distance'] / 1000:.1f}km)",
                    ha="right",
                    va="center",
                    fontsize=8,
                )

        ax.set_xlabel("Time (s)")
        ax.set_title(f"Component {component}")
        ax.set_xlim(time_range)
        ax.grid(True, alpha=0.3)

        # Add legend for the first component only
        if i == 0:
            ax.legend(loc="upper left", fontsize=8, fancybox=False)

        # Set y-axis limits to show all traces properly
        if len(stations_sorted) > 0:
            ax.set_ylim(
                -y_spacing * 0.5,
                (len(stations_sorted) - 1) * y_spacing + y_spacing * 0.5,
            )

        # Remove y-tick labels since they're just offsets
        ax.set_yticklabels([])

        # Set aspect ratio for seismogram plots to be consistent
        ax.set_aspect("auto")

    plt.savefig("OUTPUT_FILES/seismograms.png", dpi=300, bbox_inches="tight")
    print("Saved seismograms plot to OUTPUT_FILES/seismograms.png")

    plt.show(block=False)


if __name__ == "__main__":
    main()
