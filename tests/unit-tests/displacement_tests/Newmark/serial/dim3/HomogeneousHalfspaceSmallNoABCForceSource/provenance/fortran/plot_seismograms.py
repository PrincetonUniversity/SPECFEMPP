import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
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


def read_forcesolution(filename):
    """Read FORCESOLUTION file"""
    source = {}
    with open(filename, "r") as f:
        for line in f:
            if line.strip():
                if "latorUTM:" in line:
                    source["y"] = float(line.split(":")[1].strip())
                elif "longorUTM:" in line:
                    source["x"] = float(line.split(":")[1].strip())
                elif "depth:" in line:
                    source["depth"] = float(line.split(":")[1].strip())
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
    source = read_forcesolution("DATA/FORCESOLUTION")

    # Calculate epicentral distances and sort stations
    for station in stations:
        station["distance"] = calculate_epicentral_distance(
            station["x"], station["y"], source["x"], source["y"]
        )

    stations_sorted = sorted(stations, key=lambda x: x["distance"])

    # Create figure with gridspec for 1 row x 4 columns
    fig = plt.figure(figsize=(20, 6))
    gs = gridspec.GridSpec(1, 4, width_ratios=[2, 1, 1, 1], hspace=0.3, wspace=0.3)

    # Subplot 1: Source-Station geometry with circular grid (larger)
    ax1 = fig.add_subplot(gs[0, 0])

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

    # Find seismogram files
    seismogram_files = {
        "BXX": sorted(glob.glob("OUTPUT_FILES/*.BXX.semd")),
        "BXY": sorted(glob.glob("OUTPUT_FILES/*.BXY.semd")),
        "BXZ": sorted(glob.glob("OUTPUT_FILES/*.BXZ.semd")),
    }

    # Read all seismograms and find common time range
    all_seismograms = {}
    time_range = None
    max_displacement = 0

    for component in ["BXX", "BXY", "BXZ"]:
        all_seismograms[component] = {}
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

    # Plot seismograms for each component
    components = ["BXX", "BXY", "BXZ"]

    for i, component in enumerate(components):
        ax = fig.add_subplot(gs[0, i + 1])

        # Plot seismograms sorted by epicentral distance
        y_spacing = max_displacement * 2.5

        for j, station in enumerate(stations_sorted):
            station_name = station["station"]
            if station_name in all_seismograms[component]:
                time, displacement = all_seismograms[component][station_name]

                # Normalize and offset displacement
                normalized_disp = displacement / max_displacement * y_spacing * 0.8
                y_pos = j * y_spacing

                ax.plot(time, normalized_disp + y_pos, "k-", linewidth=0.8)

                # Add station label and distance
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

    plt.tight_layout()
    plt.savefig("seismogram_plot.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
