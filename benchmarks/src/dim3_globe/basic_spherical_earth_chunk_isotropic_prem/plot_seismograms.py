"""Compare SPECFEM++ and SPECFEM3D_GLOBE seismograms for the Bolivia 1994 benchmark.

Run from the benchmark build directory (cwd). Inputs, relative to cwd:

* ``DATA/STATIONS`` and ``DATA/CMTSOLUTION``
* reference (SPECFEM3D_GLOBE):
  ``reference_seismograms/{NET}.{STA}.{B}X{E,N,Z}.sem.sac``
* SPECFEM++: ``OUTPUT_FILES/results/{NET}.{STA}.S3.{B}X{E,N,Z}.sac``

Both sides are assumed to use the centroid time as t = 0 (SAC ``b`` is the
start time relative to it). Works with only the reference side present.
Outputs go to ``OUTPUT_FILES/``: ``comparison.txt``, ``{E,N,Z}_record_section.png``
and ``geometry.png``.
"""

import argparse
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import obspy  # noqa: E402

COMPONENTS = ("E", "N", "Z")
CHUNK_HALF_WIDTH_DEG = 45.0
EARTH_RADIUS_KM = 6371.0


def read_stations(path):
    """Read a SPECFEM STATIONS file into a list of dicts."""
    stations = []
    with open(path) as stations_file:
        for line in stations_file:
            fields = line.split()
            if len(fields) < 4 or line.lstrip().startswith("#"):
                continue
            stations.append(
                {
                    "station": fields[0],
                    "network": fields[1],
                    "latitude": float(fields[2]),
                    "longitude": float(fields[3]),
                }
            )
    return stations


def read_event(path):
    """Read event name, latitude, longitude and depth (km) from a CMTSOLUTION."""
    event = {"name": "event"}
    with open(path) as cmt_file:
        for line in cmt_file:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().lower()
            if key == "event name":
                event["name"] = value.strip()
            elif key in ("latitude", "longitude", "depth"):
                event[key] = float(value.split()[0])
    return event


def epicentral_distance_deg(lat1, lon1, lat2, lon2):
    """Great-circle distance in degrees (haversine on a sphere)."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return float(np.degrees(2 * np.arcsin(np.sqrt(min(1.0, a)))))


def find_trace(pattern):
    """Return the first SAC trace matching a glob pattern, or None."""
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    return obspy.read(matches[0], format="SAC")[0]


def time_axis(trace):
    """Time axis in seconds relative to the SAC reference time."""
    return (
        trace.stats.sac.get("b", 0.0) + np.arange(trace.stats.npts) * trace.stats.delta
    )


def load_pair(station, component, results_dir, reference_dir):
    """Load (specfem++, reference) traces for one station and component."""
    net, sta = station["network"], station["station"]
    ours = find_trace(os.path.join(results_dir, f"{net}.{sta}.S3.?X{component}.sac"))
    reference = find_trace(
        os.path.join(reference_dir, f"{net}.{sta}.?X{component}.sem.sac")
    )
    return ours, reference


def compare_traces(ours, reference):
    """Return (misfit, correlation) over the common window, or (None, None)."""
    t_ours, t_ref = time_axis(ours), time_axis(reference)
    window = (t_ours >= t_ref[0]) & (t_ours <= t_ref[-1])
    if window.sum() < 2:
        return None, None
    u = ours.data[window].astype(float)
    r = np.interp(t_ours[window], t_ref, reference.data.astype(float))
    norm_r = np.linalg.norm(r)
    misfit = np.linalg.norm(u - r) / norm_r if norm_r > 0 else np.nan
    norm_u = np.linalg.norm(u - u.mean()) * np.linalg.norm(r - r.mean())
    correlation = (
        float(np.dot(u - u.mean(), r - r.mean()) / norm_u) if norm_u > 0 else np.nan
    )
    return float(misfit), correlation


def describe_trace(label, trace):
    """One-line description of a trace's SAC start-time information."""
    return (
        f"{label}: b={trace.stats.sac.get('b', float('nan')):.3f} s, "
        f"delta={trace.stats.delta} s, npts={trace.stats.npts}, "
        f"starttime={trace.stats.starttime}"
    )


def format_value(value):
    """Format a float or 'n/a'."""
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.5f}"


def write_comparison(rows, header_lines, path):
    """Write and print the comparison table; return the misfits."""
    lines = list(header_lines)
    lines.append(
        f"{'NET':<4} {'STA':<8} {'COMP':<5} {'DIST_DEG':>9} {'MISFIT':>10} {'CORR':>10}"
    )
    misfits = []
    for row in sorted(
        rows, key=lambda r: (r["distance"], r["station"], r["component"])
    ):
        lines.append(
            f"{row['network']:<4} {row['station']:<8} {row['component']:<5} "
            f"{row['distance']:>9.3f} {format_value(row['misfit']):>10} "
            f"{format_value(row['correlation']):>10}"
        )
        if row["misfit"] is not None and np.isfinite(row["misfit"]):
            misfits.append(row["misfit"])
    if misfits:
        lines.append(
            f"misfit over {len(misfits)} traces: min={min(misfits):.5f} "
            f"max={max(misfits):.5f} mean={np.mean(misfits):.5f}"
        )
    else:
        lines.append("no traces with both sides present; no misfit statistics")
    text = "\n".join(lines)
    with open(path, "w") as out_file:
        out_file.write(text + "\n")
    print(text)
    return misfits


def group_labels(entries, tolerance_deg=0.4):
    """Merge labels of stations closer than tolerance into 'A / B'."""
    groups = []
    for station, _ in sorted(entries, key=lambda e: e[0]["distance"]):
        name = f"{station['network']}.{station['station']}"
        if groups and station["distance"] - groups[-1][-1][0] < tolerance_deg:
            groups[-1].append((station["distance"], name))
        else:
            groups.append([(station["distance"], name)])
    return [(np.mean([d for d, _ in g]), " / ".join(n for _, n in g)) for g in groups]


def plot_record_section(stations, traces, component, output_path):
    """Record section for one component, traces offset by epicentral distance."""
    entries = [
        (s, traces[(s["network"], s["station"], component)])
        for s in stations
        if (s["network"], s["station"], component) in traces
    ]
    entries = [
        (s, pair) for s, pair in entries if pair[0] is not None or pair[1] is not None
    ]
    if not entries:
        print(f"No traces for component {component}; skipping record section")
        return

    sides = [t for _, pair in entries for t in pair if t is not None]
    t_min = min(time_axis(t)[0] for t in sides)
    t_max = max(time_axis(t)[-1] for t in sides)
    distances = np.array([s["distance"] for s, _ in entries])
    span = max(distances.max() - distances.min(), 1.0)
    height = 0.8 * span / len(entries)

    fig, ax = plt.subplots(figsize=(9, 10))
    for station, (ours, reference) in entries:
        pair_amplitude = (
            max(np.abs(t.data).max() for t in (ours, reference) if t is not None) or 1.0
        )
        if ours is not None:
            ax.plot(
                time_axis(ours),
                station["distance"] + ours.data / pair_amplitude * height,
                "k-",
                linewidth=0.8,
                label="specfem++",
            )
        if reference is not None:
            ax.plot(
                time_axis(reference),
                station["distance"] + reference.data / pair_amplitude * height,
                "r--",
                linewidth=0.8,
                alpha=0.7,
                label="specfem3d_globe",
            )
    for distance, label in group_labels(entries):
        ax.text(
            1.01,
            distance,
            label,
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=7,
            clip_on=False,
        )
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper right", fontsize=8)
    ax.set_xlim(t_min, t_max)
    ax.set_xlabel("Time relative to centroid (s)")
    ax.set_ylabel("Epicentral distance (deg)")
    ax.set_title(f"Component {component} (each trace normalised)")
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_geometry(stations, event, output_path):
    """Plain lat/lon map of stations, epicentre and chunk extent guide."""
    center_lat, center_lon = event["latitude"], event["longitude"]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        [s["longitude"] for s in stations],
        [s["latitude"] for s in stations],
        marker="^",
        color="tab:blue",
        label="stations",
    )
    for station in stations:
        ax.annotate(
            f"{station['network']}.{station['station']}",
            (station["longitude"], station["latitude"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
        )
    ax.scatter(
        [event["longitude"]],
        [event["latitude"]],
        marker="*",
        s=250,
        color="tab:red",
        label="epicentre",
    )
    ax.scatter(
        [center_lon], [center_lat], marker="+", s=150, color="k", label="chunk centre"
    )
    for offset in (-CHUNK_HALF_WIDTH_DEG, CHUNK_HALF_WIDTH_DEG):
        ax.axhline(center_lat + offset, color="gray", linestyle="--", linewidth=0.8)
        ax.axvline(center_lon + offset, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title(f"Event {event['name']}: source-station geometry")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--max-misfit", type=float, default=None)
    parser.add_argument("--results-dir", default="OUTPUT_FILES/results")
    parser.add_argument("--reference-dir", default="reference_seismograms")
    parser.add_argument("--output-dir", default="OUTPUT_FILES")
    return parser.parse_args(argv)


def main(argv=None):
    """Compare, tabulate and plot; return the process exit code."""
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    stations = read_stations("DATA/STATIONS")
    event = read_event("DATA/CMTSOLUTION")
    for station in stations:
        station["distance"] = epicentral_distance_deg(
            event["latitude"],
            event["longitude"],
            station["latitude"],
            station["longitude"],
        )

    traces = {}
    rows = []
    first = {"specfem++": None, "reference": None}
    for station in stations:
        for component in COMPONENTS:
            ours, reference = load_pair(
                station, component, args.results_dir, args.reference_dir
            )
            traces[(station["network"], station["station"], component)] = (
                ours,
                reference,
            )
            if ours is not None and first["specfem++"] is None:
                first["specfem++"] = describe_trace("specfem++", ours)
            if reference is not None and first["reference"] is None:
                first["reference"] = describe_trace("reference", reference)
            misfit, correlation = (None, None)
            if ours is not None and reference is not None:
                misfit, correlation = compare_traces(ours, reference)
            rows.append(
                {
                    "network": station["network"],
                    "station": station["station"],
                    "component": component,
                    "distance": station["distance"],
                    "misfit": misfit,
                    "correlation": correlation,
                }
            )

    if first["specfem++"] is None:
        print(
            f"NOTICE: no SPECFEM++ traces found in {args.results_dir}; reference only"
        )
    if first["reference"] is None:
        print(f"NOTICE: no reference traces found in {args.reference_dir}")
    header = [
        "# t=0 assumed to be the centroid time on both sides (no shift applied)",
        "# " + (first["specfem++"] or "specfem++: no traces"),
        "# " + (first["reference"] or "reference: no traces"),
    ]
    misfits = write_comparison(
        rows, header, os.path.join(args.output_dir, "comparison.txt")
    )

    for component in COMPONENTS:
        plot_record_section(
            stations,
            traces,
            component,
            os.path.join(args.output_dir, f"{component}_record_section.png"),
        )
    plot_geometry(stations, event, os.path.join(args.output_dir, "geometry.png"))

    if args.max_misfit is not None and any(m > args.max_misfit for m in misfits):
        print(f"FAIL: misfit exceeds {args.max_misfit}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
