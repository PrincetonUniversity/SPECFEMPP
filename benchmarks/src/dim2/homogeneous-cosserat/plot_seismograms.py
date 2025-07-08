#!/usr/bin/env python3

import glob
import os
import numpy as np
import obspy
import matplotlib
import matplotlib.pyplot as plt
import yaml
import sys

# Set matplotlib gui off
matplotlib.use("Agg")


def get_traces(directory, ext):
    traces = []
    files = glob.glob(directory + "/*." + ext)
    ## iterate over all seismograms
    for filename in files:
        station_name = os.path.splitext(filename)[0]
        network, station, location, channel = station_name.split("/")[-1].split(".")
        trace = np.loadtxt(filename, delimiter=" ")
        starttime = trace[0, 0]
        dt = trace[1, 0] - trace[0, 0]
        traces.append(
            obspy.Trace(
                trace[:, 1],
                {
                    "network": network,
                    "station": station,
                    "location": location,
                    "channel": channel,
                    "starttime": starttime,
                    "delta": dt,
                },
            )
        )

    stream = obspy.Stream(traces)
    return stream


def main():
    if len(sys.argv) != 2:
        print("Usage: python plot_seismograms.py <output_file>")
        sys.exit(1)

    output_file = sys.argv[1]

    # Read station coordinates
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

    # Get all streams
    streamd = get_traces("OUTPUT_FILES/results", "semd")
    streamr = get_traces("OUTPUT_FILES/results", "semr")
    streamir = get_traces("OUTPUT_FILES/results", "semir")
    streamc = get_traces("OUTPUT_FILES/results", "semc")

    # Calculate distances and create master station order
    def get_station_distance(station_name):
        if station_name in stations_info:
            station_x = stations_info[station_name]["x"]
            station_z = stations_info[station_name]["z"]
            distance = np.sqrt(
                (station_x - source_x) ** 2 + (station_z - source_z) ** 2
            )
            return distance
        return float("inf")

    # Get all unique station names and sort by distance (reverse for closest at bottom)
    all_stations = set()
    for stream in [streamd, streamr, streamir, streamc]:
        for tr in stream:
            all_stations.add(tr.stats.station)

    sorted_stations = sorted(all_stations, key=get_station_distance, reverse=True)

    # Sort traces for each component according to master station order
    def sort_traces_by_station_order(stream, component):
        traces = stream.select(component=component)
        traces_dict = {tr.stats.station: tr for tr in traces}
        return [
            traces_dict[station]
            for station in sorted_stations
            if station in traces_dict
        ]

    traces_dx = sort_traces_by_station_order(streamd, "X")
    traces_dz = sort_traces_by_station_order(streamd, "Z")
    traces_r = sort_traces_by_station_order(streamr, "Y")
    traces_ir = sort_traces_by_station_order(streamir, "Y")
    traces_c = sort_traces_by_station_order(streamc, "Y")

    # Number of stations
    n_stations = len(traces_dx)

    # Create figure with subplots: rows = stations, cols = components
    fig, axes = plt.subplots(
        n_stations,
        4,
        figsize=(16, 0.7 * n_stations),
        gridspec_kw={"wspace": 0.1, "hspace": 0.1},
    )
    if n_stations == 1:
        axes = axes.reshape(1, -1)

    # Column titles
    col_titles = ["BXX.D", "BXZ.D", "BXY.R & BXY.C", "BXY.IR"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, pad=10, loc="left")

    # Calculate column-wise maximum absolute values for consistent scaling
    col_ymaxes = []

    # Column 0: Displacement X
    col0_data = []
    for tr in traces_dx:
        col0_data.extend(tr.data)
    col_ymaxes.append(np.max(np.abs(col0_data)) if col0_data else 1.0)

    # Column 1: Displacement Z
    col1_data = []
    for tr in traces_dz:
        col1_data.extend(tr.data)
    col_ymaxes.append(np.max(np.abs(col1_data)) if col1_data else 1.0)

    # Column 2: Rotation & 0.5*Curl
    col2_data = []
    for tr_r, tr_c in zip(traces_r, traces_c):
        col2_data.extend(tr_r.data)
        col2_data.extend(0.5 * tr_c.data)
    col_ymaxes.append(np.max(np.abs(col2_data)) if col2_data else 1.0)

    # Column 3: Intrinsic Rotation
    col3_data = []
    for tr in traces_ir:
        col3_data.extend(tr.data)
    col_ymaxes.append(np.max(np.abs(col3_data)) if col3_data else 1.0)

    # Plot each station
    for station_idx in range(n_stations):
        # Plot Displacement X
        if station_idx < len(traces_dx):
            tr = traces_dx[station_idx]
            time = tr.times()
            normalized_data = tr.data / col_ymaxes[0]
            axes[station_idx, 0].plot(
                time, normalized_data, "k-", linewidth=0.8, clip_on=False
            )

        # Plot Displacement Z
        if station_idx < len(traces_dz):
            tr = traces_dz[station_idx]
            time = tr.times()
            normalized_data = tr.data / col_ymaxes[1]
            axes[station_idx, 1].plot(
                time, normalized_data, "k-", linewidth=0.8, clip_on=False
            )

        # Plot Rotation (solid) & 0.5 * curl (dashed)
        if station_idx < len(traces_r) and station_idx < len(traces_c):
            tr_r = traces_r[station_idx]
            tr_c = traces_c[station_idx]
            time_r = tr_r.times()
            time_c = tr_c.times()
            normalized_r = tr_r.data / col_ymaxes[2]
            normalized_c = (0.5 * tr_c.data) / col_ymaxes[2]
            axes[station_idx, 2].plot(
                time_r,
                normalized_r,
                "k-",
                linewidth=0.8,
                label="Rotation" if station_idx == 0 else "",
                clip_on=False,
            )
            axes[station_idx, 2].plot(
                time_c,
                normalized_c,
                "r--",
                linewidth=1.0,
                alpha=0.8,
                label="$0.5\\nabla \\times \\mathbf{s}$" if station_idx == 0 else "",
                clip_on=False,
            )
            if station_idx == 0:
                axes[station_idx, 2].legend(
                    frameon=False, bbox_to_anchor=(1.0, 1.0), loc="lower right"
                )

        # Plot Intrinsic Rotation
        if station_idx < len(traces_ir):
            tr = traces_ir[station_idx]
            time = tr.times()
            normalized_data = tr.data / col_ymaxes[3]
            axes[station_idx, 3].plot(
                time,
                normalized_data,
                "k-",
                linewidth=0.8,
                label="Intrinsic Rotation" if station_idx == 0 else "",
                clip_on=False,
            )
            if station_idx == 0:
                axes[station_idx, 3].legend(
                    frameon=False, bbox_to_anchor=(1.0, 1.0), loc="lower right"
                )

        # Set y-axis labels for first column only
        axes[station_idx, 0].set_ylabel(f"{traces_dx[station_idx].stats.station}")

        # Set y-limits using column-wise scaling and add scale labels only to first row
        for col in range(4):
            axes[station_idx, col].grid(
                True, alpha=1.0, linestyle="-", linewidth=0.5, axis="x", zorder=-1
            )
            axes[station_idx, col].spines["top"].set_visible(False)
            axes[station_idx, col].spines["right"].set_visible(False)
            axes[station_idx, col].spines["left"].set_linewidth(0.8)
            axes[station_idx, col].spines["bottom"].set_visible(False)
            # axes[station_idx, col].spines['bottom'].set_linewidth(0.8)

            # Set consistent normalized y-limits for entire column
            axes[station_idx, col].set_ylim(-1.0, 1.0)

            # Set x-limits to trace times
            if col == 0 and station_idx < len(traces_dx):
                time_range = traces_dx[station_idx].times()
                axes[station_idx, col].set_xlim(time_range[0], time_range[-1])
            elif col == 1 and station_idx < len(traces_dz):
                time_range = traces_dz[station_idx].times()
                axes[station_idx, col].set_xlim(time_range[0], time_range[-1])
            elif col == 2 and station_idx < len(traces_r):
                time_range = traces_r[station_idx].times()
                axes[station_idx, col].set_xlim(time_range[0], time_range[-1])
            elif col == 3 and station_idx < len(traces_ir):
                time_range = traces_ir[station_idx].times()
                axes[station_idx, col].set_xlim(time_range[0], time_range[-1])

            # Add scale label only in first row
            if station_idx == 0:
                ymax = col_ymaxes[col]
                axes[station_idx, col].text(
                    0.02,
                    0.98,
                    f"{ymax:.2e}",
                    transform=axes[station_idx, col].transAxes,
                    verticalalignment="top",
                )

    # Remove x-tick labels from all rows except bottom
    for station_idx in range(n_stations):  # All except last row
        if station_idx < n_stations - 1:
            for col in range(4):
                axes[station_idx, col].tick_params(
                    axis="x", labelbottom=False, bottom=False
                )
        else:
            for col in range(4):
                axes[station_idx, col].tick_params(
                    axis="x", labelbottom=True, bottom=False
                )

    # Remove y-tick labels from all subplots
    for station_idx in range(n_stations):
        for col in range(4):  # All columns
            axes[station_idx, col].tick_params(axis="y", labelleft=False)

    # Set x-axis labels only for bottom row
    for col in range(4):
        axes[-1, col].set_xlabel("Time (s)")

    plt.savefig(
        output_file, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    plt.close()


if __name__ == "__main__":
    main()
