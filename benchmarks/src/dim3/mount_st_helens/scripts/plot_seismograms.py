import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# One figure per field; extension is the SPECFEM seismogram suffix.
FIELDS = [
    ("Displacement", "semd"),
    ("Velocity", "semv"),
    ("Acceleration", "sema"),
]

# Column title and SEED channel orientation. The mesh is UTM zone 10, so
# SPECFEM++ now writes geographic E/N/Z channels directly -- the same
# orientation letters as the SPECFEM3D reference traces, so no X/Y/Z -> E/N/Z
# remapping is needed.
COMPONENTS = [
    ("East (E)", "HXE"),
    ("North (N)", "HXN"),
    ("Vertical (Z)", "HXZ"),
]

DEG_KM = 111.195  # km per degree of latitude


def read_stations(filename):
    stations = []
    with open(filename, "r") as f:
        for line in f:
            if line.strip():
                parts = line.split()
                stations.append(
                    {
                        "station": parts[0],
                        "network": parts[1],
                        "lat": float(parts[2]),
                        "lon": float(parts[3]),
                    }
                )
    return stations


def read_source_lonlat(filename):
    lat = lon = None
    with open(filename, "r") as f:
        for line in f:
            if line.lower().startswith("latitude:"):
                lat = float(line.split(":", 1)[1])
            elif line.lower().startswith("longitude:"):
                lon = float(line.split(":", 1)[1])
    return lon, lat


def epicentral_distance_km(lat, lon, src_lat, src_lon):
    dlat = (lat - src_lat) * DEG_KM
    dlon = (lon - src_lon) * DEG_KM * np.cos(np.radians(src_lat))
    return np.hypot(dlat, dlon)


def read_trace(filename):
    if not os.path.isfile(filename):
        return None
    data = np.loadtxt(filename)
    return data[:, 0], data[:, 1]


def plot_field(
    field_name, ext, stations, src_lon, src_lat, results_dir, ref_dir, out_path
):
    network = stations[0]["network"]

    # Collect every trace first so we can use a single amplitude scale shared
    # across all stations and all 3 components (preserves relative amplitudes).
    traces = {}  # (station, comp_idx) -> dict(computed=..., reference=...)
    max_amp = 0.0
    time_range = None
    for station in stations:
        for ci, (_title, comp) in enumerate(COMPONENTS):
            computed = read_trace(
                os.path.join(
                    results_dir, f"{network}.{station['station']}.S3.{comp}.{ext}"
                )
            )
            reference = read_trace(
                os.path.join(ref_dir, f"{network}.{station['station']}.{comp}.{ext}")
            )
            traces[(station["station"], ci)] = {
                "computed": computed,
                "reference": reference,
            }
            for tr in (computed, reference):
                if tr is not None:
                    max_amp = max(max_amp, np.abs(tr[1]).max())
                    t0, t1 = tr[0].min(), tr[0].max()
                    time_range = (
                        (t0, t1)
                        if time_range is None
                        else (min(time_range[0], t0), max(time_range[1], t1))
                    )
    if max_amp == 0.0:
        max_amp = 1.0

    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(1, 4, width_ratios=[2, 1, 1, 1], wspace=0.3)

    ax_map = fig.add_subplot(gs[0, 0])
    for station in stations:
        ax_map.plot(station["lon"], station["lat"], "kv", markersize=8)
        ax_map.text(
            station["lon"],
            station["lat"] + 0.005,
            station["station"],
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax_map.plot(src_lon, src_lat, "r*", markersize=16, label="Source")
    ax_map.set_xlabel("Longitude (deg)")
    ax_map.set_ylabel("Latitude (deg)")
    ax_map.set_title(f"{field_name}\nSource-Station Geometry")
    ax_map.legend()
    ax_map.grid(True, alpha=0.3)

    y_spacing = max_amp * 2.5
    for ci, (title, _comp) in enumerate(COMPONENTS):
        ax = fig.add_subplot(gs[0, ci + 1])
        for j, station in enumerate(stations):
            tr = traces[(station["station"], ci)]
            offset = j * y_spacing
            if tr["reference"] is not None:
                t, v = tr["reference"]
                ax.plot(t, v / max_amp * y_spacing * 0.8 + offset, "r--", linewidth=0.9)
            if tr["computed"] is not None:
                t, v = tr["computed"]
                ax.plot(t, v / max_amp * y_spacing * 0.8 + offset, "k-", linewidth=0.8)
            ax.text(
                time_range[0] if time_range else 0,
                offset,
                f"{station['station']}\n({station['distance']:.1f} km)",
                ha="right",
                va="center",
                fontsize=7,
            )
        ax.set_xlabel("Time (s)")
        ax.set_title(title)
        if time_range:
            ax.set_xlim(time_range)
            ax.set_ylim(
                -y_spacing * 0.5, (len(stations) - 1) * y_spacing + y_spacing * 0.5
            )
        ax.grid(True, alpha=0.3)
        ax.set_yticklabels([])
        if ci == 0:
            ax.plot([], [], "k-", label="SPECFEM++")
            ax.plot([], [], "r--", label="SPECFEM3D (ref)")
            ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"Mount St. Helens — {field_name}\n"
        f"normalized to peak amplitude = {max_amp:.3e} (shared across all 3 components)",
        fontsize=14,
    )
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {out_path}")


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stations = read_stations(os.path.join(base_dir, "DATA", "STATIONS"))
    src_lon, src_lat = read_source_lonlat(os.path.join(base_dir, "DATA", "CMTSOLUTION"))

    for station in stations:
        station["distance"] = epicentral_distance_km(
            station["lat"], station["lon"], src_lat, src_lon
        )
    stations.sort(key=lambda s: s["distance"])

    results_dir = os.path.join(base_dir, "OUTPUT_FILES", "results")
    ref_dir = os.path.join(base_dir, "reference_seismograms")
    out_dir = os.path.join(base_dir, "OUTPUT_FILES")

    for field_name, ext in FIELDS:
        out_path = os.path.join(out_dir, f"seismogram_{ext}.png")
        plot_field(
            field_name, ext, stations, src_lon, src_lat, results_dir, ref_dir, out_path
        )


if __name__ == "__main__":
    main()
