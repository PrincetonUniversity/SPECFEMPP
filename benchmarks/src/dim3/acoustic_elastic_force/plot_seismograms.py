import glob
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


def read_stations(filename):
    """Read STATIONS file."""
    stations = []
    with open(filename, "r") as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                stations.append(
                    {
                        "station": parts[0],
                        "network": parts[1],
                        "x": float(parts[3]),
                        "y": float(parts[2]),
                        "elevation": float(parts[4]),
                        "burial": float(parts[5]),
                    }
                )
    return stations


def read_source(filename):
    """Read source coordinates from source YAML file."""
    source = {}
    with open(filename, "r") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if ":" not in line:
                continue

            key, value = [part.strip() for part in line.split(":", 1)]
            if key in ("x", "y", "z") and key not in source:
                source[key] = float(value)

            if all(key in source for key in ("x", "y", "z")):
                break

    missing_keys = [key for key in ("x", "y", "z") if key not in source]
    if missing_keys:
        raise ValueError(f"Missing source coordinate(s): {', '.join(missing_keys)}")

    return source


def read_seismogram(filename):
    """Read seismogram file."""
    data = np.loadtxt(filename)
    return data[:, 0], data[:, 1]


def calculate_epicentral_distance(station_x, station_y, source_x, source_y):
    """Calculate epicentral distance."""
    return np.sqrt((station_x - source_x) ** 2 + (station_y - source_y) ** 2)


def station_name_from_path(filename):
    """Extract the station name from a SPECFEM seismogram filename."""
    return Path(filename).name.split(".")[1]


def load_seismograms(pattern):
    """Load all seismograms matching pattern by station name."""
    seismograms = {}
    for filename in sorted(glob.glob(pattern)):
        station_name = station_name_from_path(filename)
        seismograms[station_name] = read_seismogram(filename)
    return seismograms


def plot_geometry(stations, source):
    """Plot source-station geometry."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    for station in stations:
        ax.plot(
            station["x"],
            station["y"],
            "rv",
            markersize=8,
            label="Stations" if station == stations[0] else "",
        )
        ax.text(
            station["x"],
            station["y"] + 1000,
            station["station"],
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.plot(source["x"], source["y"], "r*", markersize=15, label="Source")

    max_dist = max(station["distance"] for station in stations)
    for radius in [5000, 10000, 20000, 30000, 40000, 50000]:
        if radius <= max_dist * 1.2:
            circle = Circle(
                (source["x"], source["y"]),
                radius,
                fill=False,
                linestyle="--",
                alpha=0.3,
                color="gray",
            )
            ax.add_patch(circle)
            ax.text(
                source["x"] + radius * 0.7,
                source["y"] + radius * 0.7,
                f"{radius / 1000:.0f}km",
                fontsize=8,
                alpha=0.7,
            )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Source-Station Geometry")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.savefig("OUTPUT_FILES/geometry.png", dpi=300, bbox_inches="tight")
    print("Saved source-station geometry plot to OUTPUT_FILES/geometry.png")


def plot_component_group(
    stations_sorted, component_group, reference_component_group, output_filename
):
    """Plot a family of seismograms sorted by epicentral distance."""
    components = [
        component
        for component, seismograms in component_group.items()
        if len(seismograms) > 0 or len(reference_component_group[component]) > 0
    ]
    if not components:
        print(f"No seismograms found for {output_filename}")
        return

    time_range = None
    max_amplitude = 0.0
    for component in components:
        all_seismograms = list(component_group[component].values()) + list(
            reference_component_group[component].values()
        )
        for time, amplitude in all_seismograms:
            if time_range is None:
                time_range = (time.min(), time.max())
            else:
                time_range = (
                    min(time_range[0], time.min()),
                    max(time_range[1], time.max()),
                )
            max_amplitude = max(max_amplitude, np.abs(amplitude).max())

    if max_amplitude == 0.0:
        max_amplitude = 1.0

    fig = plt.figure(figsize=(5 * len(components), 6))
    gs = gridspec.GridSpec(1, len(components), hspace=0.3, wspace=0.3)
    y_spacing = max_amplitude * 2.5

    for i, component in enumerate(components):
        ax = fig.add_subplot(gs[0, i])
        seismograms = component_group[component]
        reference_seismograms = reference_component_group[component]

        for j, station in enumerate(stations_sorted):
            station_name = station["station"]
            y_pos = j * y_spacing

            if station_name in seismograms:
                time, amplitude = seismograms[station_name]
                normalized_amplitude = amplitude / max_amplitude * y_spacing * 0.8

                ax.plot(
                    time,
                    normalized_amplitude + y_pos,
                    "k-",
                    linewidth=0.8,
                    label="specfem++" if j == 0 else "",
                )

            if station_name in reference_seismograms:
                time_ref, amplitude_ref = reference_seismograms[station_name]
                normalized_amplitude_ref = (
                    amplitude_ref / max_amplitude * y_spacing * 0.8
                )

                ax.plot(
                    time_ref,
                    normalized_amplitude_ref + y_pos,
                    "r--",
                    linewidth=0.8,
                    alpha=0.7,
                    label="xspecfem3D" if j == 0 else "",
                )

            if station_name in seismograms or station_name in reference_seismograms:
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
        if i == 0:
            ax.legend(loc="upper left", fontsize=8, fancybox=False)
        ax.set_ylim(
            -y_spacing * 0.5,
            (len(stations_sorted) - 1) * y_spacing + y_spacing * 0.5,
        )
        ax.set_yticklabels([])
        ax.set_aspect("auto")

    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"Saved seismogram plot to {output_filename}")


def main():
    stations = read_stations("DATA/STATIONS")
    source = read_source("source.yaml")

    for station in stations:
        station["distance"] = calculate_epicentral_distance(
            station["x"], station["y"], source["x"], source["y"]
        )

    stations_sorted = sorted(stations, key=lambda station: station["distance"])
    plot_geometry(stations, source)

    displacement = {
        "MXX": load_seismograms("OUTPUT_FILES/results/*.S3.MXX.semd"),
        "MXY": load_seismograms("OUTPUT_FILES/results/*.S3.MXY.semd"),
        "MXZ": load_seismograms("OUTPUT_FILES/results/*.S3.MXZ.semd"),
    }
    reference_displacement = {
        "MXX": load_seismograms("reference_seismograms/*.S3.MXX.semd"),
        "MXY": load_seismograms("reference_seismograms/*.S3.MXY.semd"),
        "MXZ": load_seismograms("reference_seismograms/*.S3.MXZ.semd"),
    }
    velocity = {
        "MXX": load_seismograms("OUTPUT_FILES/results/*.S3.MXX.semv"),
        "MXY": load_seismograms("OUTPUT_FILES/results/*.S3.MXY.semv"),
        "MXZ": load_seismograms("OUTPUT_FILES/results/*.S3.MXZ.semv"),
    }
    reference_velocity = {
        "MXX": load_seismograms("reference_seismograms/*.S3.MXX.semv"),
        "MXY": load_seismograms("reference_seismograms/*.S3.MXY.semv"),
        "MXZ": load_seismograms("reference_seismograms/*.S3.MXZ.semv"),
    }
    acceleration = {
        "MXX": load_seismograms("OUTPUT_FILES/results/*.S3.MXX.sema"),
        "MXY": load_seismograms("OUTPUT_FILES/results/*.S3.MXY.sema"),
        "MXZ": load_seismograms("OUTPUT_FILES/results/*.S3.MXZ.sema"),
    }
    reference_acceleration = {
        "MXX": load_seismograms("reference_seismograms/*.S3.MXX.sema"),
        "MXY": load_seismograms("reference_seismograms/*.S3.MXY.sema"),
        "MXZ": load_seismograms("reference_seismograms/*.S3.MXZ.sema"),
    }
    pressure = {
        "MXP": load_seismograms("OUTPUT_FILES/results/*.S3.MXP.semp"),
    }
    reference_pressure = {
        "MXP": load_seismograms("reference_seismograms/*.S3.MXP.semp"),
    }

    plot_component_group(
        stations_sorted,
        displacement,
        reference_displacement,
        "OUTPUT_FILES/displacement_seismograms.png",
    )
    plot_component_group(
        stations_sorted,
        velocity,
        reference_velocity,
        "OUTPUT_FILES/velocity_seismograms.png",
    )
    plot_component_group(
        stations_sorted,
        acceleration,
        reference_acceleration,
        "OUTPUT_FILES/acceleration_seismograms.png",
    )
    plot_component_group(
        stations_sorted,
        pressure,
        reference_pressure,
        "OUTPUT_FILES/pressure_seismograms.png",
    )

    plt.show(block=False)


if __name__ == "__main__":
    main()
