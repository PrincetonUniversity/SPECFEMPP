#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys


def main():
    if len(sys.argv) != 2:
        print("Usage: python plot_geometry.py <output_file>")
        sys.exit(1)

    output_file = sys.argv[1]

    # Read station coordinates
    stations = []
    station_names = []
    with open("OUTPUT_FILES/STATIONS", "r") as f:
        for line in f:
            if line.strip():
                parts = line.split()
                station_name = parts[0]
                x_coord = float(parts[2])
                z_coord = float(parts[3])
                stations.append([x_coord, z_coord])
                station_names.append(station_name)

    stations = np.array(stations)

    # Read source coordinates
    with open("source.yaml", "r") as f:
        source_data = yaml.safe_load(f)
    source_x = source_data["sources"][0]["force"]["x"]
    source_z = source_data["sources"][0]["force"]["z"]

    # Read topography data
    try:
        topo_data = np.loadtxt("topography.dat")
        topo_x = topo_data[:, 0]
        topo_z = topo_data[:, 1]
    except Exception as e:
        print(f"Warning: Could not read topography.dat, skipping topography ({e})")
        topo_x = None
        topo_z = None

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Plot topography if available
    if topo_x is not None and topo_z is not None:
        ax.plot(topo_x, topo_z, "k-", linewidth=2, label="Topography")
        # Fill below topography to show surface
        ax.fill_between(topo_x, topo_z, min(topo_z) - 500, color="lightgray", alpha=0.3)

    # Calculate distances from source for color coding
    distances = np.sqrt(
        (stations[:, 0] - source_x) ** 2 + (stations[:, 1] - source_z) ** 2
    )

    # Plot stations with color coding by distance
    scatter = ax.scatter(
        stations[:, 0],
        stations[:, 1],
        c=distances,
        cmap="viridis_r",
        s=100,
        marker="^",
        edgecolors="black",
        linewidth=1,
        label="Stations",
        zorder=5,
    )

    # Add station labels
    for i, name in enumerate(station_names):
        ax.annotate(
            name,
            (stations[i, 0], stations[i, 1]),
            xytext=(5, 5),
            textcoords="offset points",
        )

    # Plot source
    ax.scatter(
        source_x,
        source_z,
        s=200,
        marker="*",
        color="red",
        edgecolors="black",
        linewidth=0.5,
        label="Source",
        zorder=10,
    )

    # Source label removed per request

    # Add colorbar for station distances
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=30)
    cbar.set_label("Distance from Source (m)")

    # Set labels
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")

    # Set equal aspect ratio
    ax.set_aspect("equal")

    # Add grid
    ax.grid(True, alpha=0.3)

    # Add legend
    ax.legend(
        loc="upper right", frameon=True, fancybox=False, shadow=False, framealpha=1.0
    )

    # Invert y-axis to match typical seismic convention (depth increases downward)
    ax.invert_yaxis()

    # Add domain boundaries
    domain_x = [0, 4000, 4000, 0, 0]
    domain_z = [0, 0, 4000, 4000, 0]
    ax.plot(domain_x, domain_z, "k--", linewidth=2, alpha=0.7, label="Domain Boundary")

    # Set axes limits to show entire domain (0 to 4000 in both directions)
    domain_padding = 200  # Add small padding around domain
    ax.set_xlim(-domain_padding, 4000 + domain_padding)
    ax.set_ylim(-domain_padding, 4000 + domain_padding)

    # Add distance circles around source
    circle_distances = [500, 1000, 1500]  # meters
    for dist in circle_distances:
        circle = plt.Circle(
            (source_x, source_z),
            dist,
            fill=False,
            linestyle="--",
            alpha=0.5,
            color="gray",
        )
        ax.add_patch(circle)
        # Add distance labels
        ax.text(
            source_x + dist * 0.7,
            source_z - dist * 0.7,
            f"{dist}m",
            alpha=0.7,
            bbox=dict(boxstyle="square,pad=0.2", facecolor="white", alpha=1.0),
        )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Geometry plot saved to: {output_file}")


if __name__ == "__main__":
    main()
