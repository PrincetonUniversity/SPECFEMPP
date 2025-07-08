#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys
import os


def main():
    if len(sys.argv) != 2:
        print("Usage: python plot_reference_comparison.py <output_file>")
        sys.exit(1)

    output_file = sys.argv[1]

    # Read station coordinates for distance-based sorting
    stations_info = {}
    with open("OUTPUT_FILES/STATIONS", "r") as f:
        for line in f:
            if line.strip():
                parts = line.split()
                station_name = parts[0]
                network = parts[1]
                x_coord = float(parts[2])
                z_coord = float(parts[3])
                stations_info[station_name] = {
                    "network": network,
                    "x": x_coord,
                    "z": z_coord,
                }

    # Read source coordinates
    with open("source.yaml", "r") as f:
        source_data = yaml.safe_load(f)
    source_x = source_data["sources"][0]["force"]["x"]
    source_z = source_data["sources"][0]["force"]["z"]

    # Calculate distances and create station order
    def get_station_distance(station_name):
        if station_name in stations_info:
            station_x = stations_info[station_name]["x"]
            station_z = stations_info[station_name]["z"]
            distance = np.sqrt(
                (station_x - source_x) ** 2 + (station_z - source_z) ** 2
            )
            return distance
        return float("inf")

    # Get station names and sort by distance (reverse for closest at bottom)
    station_names = [f"S000{i + 1}" for i in range(8)]
    sorted_stations = sorted(station_names, key=get_station_distance, reverse=True)

    # Load displacement data (with spin)
    print("Loading displacement data...")

    # Load reference data
    ux_ref_orig = np.load("reference/traces_fd/ux.npy")[:, ::5]  # Subsample by 5
    uz_ref_orig = np.load("reference/traces_fd/uz.npy")[:, ::5]

    # Reorder reference data to match sorted stations
    ux_ref = np.zeros(ux_ref_orig.shape)
    uz_ref = np.zeros(uz_ref_orig.shape)

    # Load SPECFEM data
    ux_specfem = np.zeros(ux_ref.shape)
    uz_specfem = np.zeros(uz_ref.shape)

    for i, station in enumerate(sorted_stations):
        station_idx = int(station[-1]) - 1  # Convert S0001 -> 0, S0002 -> 1, etc.

        # Reorder reference data to match sorted stations
        ux_ref[i, :] = ux_ref_orig[station_idx, :]
        uz_ref[i, :] = uz_ref_orig[station_idx, :]

        # Load X displacement
        file_x = f"OUTPUT_FILES/results/AA.{station}.S2.BXX.semd"
        if os.path.exists(file_x):
            ux_specfem[i, :] = np.loadtxt(file_x)[:, 1]

        # Load Z displacement
        file_z = f"OUTPUT_FILES/results/AA.{station}.S2.BXZ.semd"
        if os.path.exists(file_z):
            uz_specfem[i, :] = np.loadtxt(file_z)[:, 1]

    # Load rotation data
    print("Loading rotation data...")

    # Load reference rotation data
    ry_ref_orig = np.load("reference/traces_fd/ry.npy")[:, ::5]

    # Reorder reference data to match sorted stations
    ry_ref = np.zeros(ry_ref_orig.shape)

    # Load SPECFEM rotation data
    ry_specfem = np.zeros(ry_ref.shape)

    for i, station in enumerate(sorted_stations):
        station_idx = int(station[-1]) - 1  # Convert S0001 -> 0, S0002 -> 1, etc.

        # Reorder reference data to match sorted stations
        ry_ref[i, :] = ry_ref_orig[station_idx, :]

        file_r = f"OUTPUT_FILES/results/AA.{station}.S2.BXY.semr"
        if os.path.exists(file_r):
            ry_specfem[i, :] = np.loadtxt(file_r)[:, 1]

    # Calculate column-wise normalization
    ux_max = max(np.abs(ux_ref).max(), np.abs(ux_specfem).max())
    uz_max = max(np.abs(uz_ref).max(), np.abs(uz_specfem).max())
    ry_max = max(np.abs(ry_ref).max(), np.abs(ry_specfem).max())

    n_stations = len(sorted_stations)

    # Create figure: 3 columns (UX, UZ, RY), n_stations rows
    fig, axes = plt.subplots(
        n_stations,
        3,
        figsize=(12, 0.7 * n_stations),
        gridspec_kw={"wspace": 0.1, "hspace": 0},
    )
    if n_stations == 1:
        axes = axes.reshape(1, -1)

    # Column titles
    col_titles = ["Displacement X", "Displacement Z", "Rotation Y"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, pad=10, loc="left")

    # Plot each station
    for station_idx in range(n_stations):
        station_name = sorted_stations[station_idx]

        # Plot Displacement X comparison
        ux_ref_norm = ux_ref[station_idx, :] / ux_max
        ux_spec_norm = ux_specfem[station_idx, :] / ux_max

        axes[station_idx, 0].plot(
            ux_ref_norm,
            "k-",
            linewidth=0.8,
            label="Reference" if station_idx == 0 else "",
            clip_on=False,
        )
        axes[station_idx, 0].plot(
            ux_spec_norm,
            "r--",
            linewidth=0.8,
            alpha=0.8,
            label="SPECFEM" if station_idx == 0 else "",
            clip_on=False,
        )

        # Plot Displacement Z comparison
        uz_ref_norm = uz_ref[station_idx, :] / uz_max
        uz_spec_norm = uz_specfem[station_idx, :] / uz_max

        axes[station_idx, 1].plot(uz_ref_norm, "k-", linewidth=0.8, clip_on=False)
        axes[station_idx, 1].plot(
            uz_spec_norm, "r--", linewidth=0.8, alpha=0.8, clip_on=False
        )

        # Plot Rotation Y comparison
        ry_ref_norm = ry_ref[station_idx, :] / ry_max
        ry_spec_norm = ry_specfem[station_idx, :] / ry_max

        axes[station_idx, 2].plot(ry_ref_norm, "k-", linewidth=0.8, clip_on=False)
        axes[station_idx, 2].plot(
            ry_spec_norm, "r--", linewidth=0.8, alpha=0.8, clip_on=False
        )

        # Add legend to first row
        if station_idx == 0:
            axes[station_idx, 0].legend(
                frameon=False, bbox_to_anchor=(1.0, 1.0), loc="lower right"
            )

        # Set y-axis labels for first column only (station names)
        axes[station_idx, 0].set_ylabel(station_name)

        # Set plot styling for all columns
        for col in range(3):
            axes[station_idx, col].grid(
                True, alpha=1.0, linestyle="-", linewidth=0.5, axis="x", zorder=-1
            )
            axes[station_idx, col].spines["top"].set_visible(False)
            axes[station_idx, col].spines["right"].set_visible(False)
            axes[station_idx, col].spines["left"].set_linewidth(0.8)
            axes[station_idx, col].spines["bottom"].set_visible(False)

            # Set normalized y-limits
            axes[station_idx, col].set_ylim(-1.0, 1.0)

            # Add scale label only in first row
            if station_idx == 0:
                if col == 0:
                    scale_val = ux_max
                elif col == 1:
                    scale_val = uz_max
                else:
                    scale_val = ry_max

                axes[station_idx, col].text(
                    0.02,
                    0.98,
                    f"{scale_val:.2e}",
                    transform=axes[station_idx, col].transAxes,
                    verticalalignment="top",
                )

    # Remove x-tick labels from all rows except bottom
    for station_idx in range(n_stations):
        if station_idx < n_stations - 1:
            for col in range(3):
                axes[station_idx, col].tick_params(
                    axis="x", labelbottom=False, bottom=False
                )
        else:
            for col in range(3):
                axes[station_idx, col].tick_params(
                    axis="x", labelbottom=True, bottom=False
                )

    # Remove y-tick labels from all subplots
    for station_idx in range(n_stations):
        for col in range(3):
            axes[station_idx, col].tick_params(axis="y", labelleft=False)

    # Set x-axis labels only for bottom row
    for col in range(3):
        axes[-1, col].set_xlabel("Sample")

    plt.savefig(
        output_file, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    plt.close()

    print(f"Reference comparison plot saved to: {output_file}")


if __name__ == "__main__":
    main()
