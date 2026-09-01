import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
from matplotlib.patches import Circle

# (row label, trace file extension) per requested seismogram type
FIELDS = [
    ("Displacement", "semd"),
    ("Rotation", "semr"),
    ("Intrinsic rotation", "semir"),
    ("Curl", "semc"),
]
COMPONENTS = ["BXX", "BXY", "BXZ"]
REFERENCE_DIRECTORIES = ("reference_seismograms",)


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


def read_source(filename):
    """Read source position from source.yaml (single cosserat-force source)"""
    source = {}
    with open(filename, "r") as f:
        for line in f:
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip().lstrip("- ").strip()
            if key in ("x", "y", "z"):
                source[key] = float(value)
    return source


def read_seismogram(filename):
    """Read seismogram file"""
    data = np.loadtxt(filename)
    return data[:, 0], data[:, 1]  # time, value


def find_reference_files(component, extension):
    """Find reference traces."""
    for directory in REFERENCE_DIRECTORIES:
        files = sorted(glob.glob(f"{directory}/*.S3.{component}.{extension}"))
        if files:
            return files
    return []


def calculate_distance(station, source):
    """Source-receiver distance (stations are buried, so include depth)"""
    return np.sqrt(
        (station["x"] - source["x"]) ** 2
        + (station["y"] - source["y"]) ** 2
        + (station["burial"] - source["z"]) ** 2
    )


def main():
    # Read station and source data
    stations = read_stations("DATA/STATIONS")
    source = read_source("source.yaml")

    # Calculate source-receiver distances and sort stations
    for station in stations:
        station["distance"] = calculate_distance(station, source)

    stations_sorted = sorted(stations, key=lambda x: x["distance"])

    # One row per field type; geometry panel spans all rows in the first column
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(
        len(FIELDS), 4, width_ratios=[1.6, 1, 1, 1], hspace=0.4, wspace=0.3
    )

    # Subplot 1: Source-Station geometry (map view) with circular grid
    ax1 = fig.add_subplot(gs[:, 0])

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
            f"{station['station']}\n(z={station['burial'] / 1000:.1f}km)",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Plot source (epicenter)
    ax1.plot(source["x"], source["y"], "r*", markersize=15, label="Source")

    # Add circular epicentral distance grid
    max_dist = max([s["distance"] for s in stations])
    circles = [2500, 5000, 7500, 10000, 15000, 20000]
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
                f"{radius / 1000:.1f}km",
                fontsize=8,
                alpha=0.7,
            )

    ax1.set_xlabel("X (UTM)")
    ax1.set_ylabel("Y (UTM)")
    ax1.set_title("Source-Station Geometry (map view)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal")

    # Plot record sections: one row per field type, one column per component
    for row, (field_name, ext) in enumerate(FIELDS):
        # Read all traces for this field type
        seismograms = {}
        reference_seismograms = {}
        time_range = None
        max_amplitude = 0

        for component in COMPONENTS:
            seismograms[component] = {}
            files = sorted(glob.glob(f"OUTPUT_FILES/results/*.S3.{component}.{ext}"))
            for filename in files:
                station_name = filename.split("/")[-1].split(".")[1]
                time, value = read_seismogram(filename)
                seismograms[component][station_name] = (time, value)

                if time_range is None:
                    time_range = (time.min(), time.max())
                else:
                    time_range = (
                        min(time_range[0], time.min()),
                        max(time_range[1], time.max()),
                    )
                max_amplitude = max(max_amplitude, np.abs(value).max())

            # Reference data is available only for displacement and rotation.
            if ext in ("semd", "semr"):
                for component in COMPONENTS:
                    reference_seismograms[component] = {}
                    for filename in find_reference_files(component, ext):
                        station_name = filename.split("/")[-1].split(".")[1]
                        time, value = read_seismogram(filename)
                        reference_seismograms[component][station_name] = (time, value)

                        if time_range is None:
                            time_range = (time.min(), time.max())
                        else:
                            time_range = (
                                min(time_range[0], time.min()),
                                max(time_range[1], time.max()),
                            )
                        max_amplitude = max(max_amplitude, np.abs(value).max())

        if max_amplitude == 0 or time_range is None:
            continue

        for i, component in enumerate(COMPONENTS):
            ax = fig.add_subplot(gs[row, i + 1])

            # Plot seismograms sorted by source-receiver distance
            y_spacing = max_amplitude * 2.5

            for j, station in enumerate(stations_sorted):
                station_name = station["station"]

                if station_name not in seismograms[component]:
                    continue

                time, value = seismograms[component][station_name]

                # Normalize (per field type) and offset
                normalized = value / max_amplitude * y_spacing * 0.8
                y_pos = j * y_spacing

                ax.plot(time, normalized + y_pos, "k-", linewidth=0.8)

                if station_name in reference_seismograms.get(component, {}):
                    time_ref, value_ref = reference_seismograms[component][station_name]
                    normalized_ref = value_ref / max_amplitude * y_spacing * 0.8
                    ax.plot(
                        time_ref,
                        normalized_ref + y_pos,
                        "r--",
                        linewidth=0.8,
                        alpha=0.7,
                        label="Analytic Solution" if j == 0 else "",
                    )

                # Add station label and distance on the leftmost column
                if i == 0:
                    ax.text(
                        time_range[0] - (time_range[1] - time_range[0]) * 0.05,
                        y_pos,
                        f"{station_name}\n({station['distance'] / 1000:.1f}km)",
                        ha="right",
                        va="center",
                        fontsize=8,
                    )

            if row == len(FIELDS) - 1:
                ax.set_xlabel("Time (s)")
            ax.set_title(f"{field_name} {component}", fontsize=10)
            ax.set_xlim(time_range)
            ax.grid(True, alpha=0.3)

            if i == 0 and ext in ("semd", "semr"):
                ax.legend(loc="upper left", fontsize=8, fancybox=False)

            # Set y-axis limits to show all traces properly
            if len(stations_sorted) > 0:
                ax.set_ylim(
                    -y_spacing * 0.5,
                    (len(stations_sorted) - 1) * y_spacing + y_spacing * 0.5,
                )

            # Remove y-tick labels since they're just offsets
            ax.set_yticklabels([])
            ax.set_aspect("auto")

    plt.tight_layout()
    plt.savefig("OUTPUT_FILES/seismogram_plot.png", dpi=300, bbox_inches="tight")
    plt.show(block=False)


if __name__ == "__main__":
    main()
