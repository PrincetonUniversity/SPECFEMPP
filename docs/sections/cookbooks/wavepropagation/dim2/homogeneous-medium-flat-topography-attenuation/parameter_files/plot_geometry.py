"""Plot the benchmark geometry: domain box, source, and station locations."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

TOPOGRAPHY_FILE = "topography.dat"
PAR_FILE = "Par_File_attenuation_on"
SOURCE_FILE = "source.yaml"
STATIONS_FILE = "OUTPUT_FILES/attenuation_on/STATIONS"
OUTPUT_FILE = "OUTPUT_FILES/results/geometry.png"


# ---------------------------------------------------------------------------
# Parse Par_File for xmin/xmax
# ---------------------------------------------------------------------------
def parse_par_file(path):
    params = {}
    with open(path) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if "=" in line:
                key, _, val = line.partition("=")
                params[key.strip()] = val.strip().split()[0]
    xmin = float(params.get("xmin", "0").rstrip("d0").rstrip("d"))
    xmax = float(params.get("xmax", "5000").rstrip("d0").rstrip("d"))
    return xmin, xmax


# ---------------------------------------------------------------------------
# Parse topography.dat — returns list of interfaces, each a list of (x,z)
# ---------------------------------------------------------------------------
def parse_topography(path):
    with open(path) as f:
        lines = [l1.split("#")[0].strip() for l1 in f]
    lines = [l2 for l2 in lines if l2]

    idx = 0
    n_interfaces = int(lines[idx])
    idx += 1
    interfaces = []
    for _ in range(n_interfaces):
        n_pts = int(lines[idx])
        idx += 1
        pts = []
        for _ in range(n_pts):
            x, z = map(float, lines[idx].split())
            idx += 1
            pts.append((x, z))
        interfaces.append(pts)
    return interfaces


# ---------------------------------------------------------------------------
# Parse source.yaml — returns list of (x, z) source positions
# ---------------------------------------------------------------------------
def parse_sources(path):
    with open(path) as f:
        doc = yaml.safe_load(f)
    sources = []
    for src in doc.get("sources", []):
        for src_type, params in src.items():
            x = float(params.get("x", 0))
            z = float(params.get("z", 0))
            sources.append((x, z))
    return sources


# ---------------------------------------------------------------------------
# Parse STATIONS file — columns: name  network  x  z  ...
# ---------------------------------------------------------------------------
def parse_stations(path):
    stations = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            name = parts[0]
            network = parts[1]
            x = float(parts[2])
            z = float(parts[3])
            stations.append((name, network, x, z))
    return stations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    xmin, xmax = parse_par_file(PAR_FILE)
    interfaces = parse_topography(TOPOGRAPHY_FILE)
    sources = parse_sources(SOURCE_FILE)
    stations = parse_stations(STATIONS_FILE)

    bottom_pts = interfaces[0]
    top_pts = interfaces[-1]

    def interp_z(pts, x):
        xs = [p[0] for p in pts]
        zs = [p[1] for p in pts]
        return float(np.interp(x, xs, zs))

    bot_xs = np.array([p[0] for p in bottom_pts])
    bot_zs = np.array([p[1] for p in bottom_pts])
    top_xs = np.array([p[0] for p in top_pts])
    top_zs = np.array([p[1] for p in top_pts])

    mask_bot = (bot_xs >= xmin) & (bot_xs <= xmax)
    mask_top = (top_xs >= xmin) & (top_xs <= xmax)

    bot_x = np.concatenate([[xmin], bot_xs[mask_bot], [xmax]])
    bot_z = np.concatenate(
        [[interp_z(bottom_pts, xmin)], bot_zs[mask_bot], [interp_z(bottom_pts, xmax)]]
    )
    top_x = np.concatenate([[xmin], top_xs[mask_top], [xmax]])
    top_z = np.concatenate(
        [[interp_z(top_pts, xmin)], top_zs[mask_top], [interp_z(top_pts, xmax)]]
    )

    domain_x = np.concatenate([bot_x, top_x[::-1], [bot_x[0]]])
    domain_z = np.concatenate([bot_z, top_z[::-1], [bot_z[0]]])

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.fill(domain_x, domain_z, color="#e8f4f8", zorder=0)
    ax.plot(domain_x, domain_z, color="steelblue", linewidth=1.5, zorder=1)

    ax.plot(top_x, top_z, color="sienna", linewidth=2, label="Free surface", zorder=2)

    side_z = [interp_z(bottom_pts, xmin), interp_z(top_pts, xmin)]
    ax.plot(
        [xmin, xmin],
        side_z,
        color="gray",
        linewidth=1.5,
        linestyle="--",
        label="Absorbing boundary",
        zorder=2,
    )
    ax.plot(
        [xmax, xmax],
        [interp_z(bottom_pts, xmax), interp_z(top_pts, xmax)],
        color="gray",
        linewidth=1.5,
        linestyle="--",
        zorder=2,
    )
    ax.plot(bot_x, bot_z, color="gray", linewidth=1.5, linestyle="--", zorder=2)

    sx = [s[2] for s in stations]
    sz = [s[3] for s in stations]
    sc = ax.scatter(
        sx, sz, marker="v", color="royalblue", s=80, zorder=5, label="Stations"
    )
    sc.set_clip_on(False)
    z_surface = interp_z(top_pts, xmin)
    surface_count = 0
    for name, network, x, z in stations:
        at_surface = abs(z - z_surface) < 1.0
        if at_surface:
            xytext = (0, 6) if surface_count % 2 == 0 else (0, -12)
            surface_count += 1
        else:
            xytext = (4, -12)
        ax.annotate(
            f"{network}.{name}",
            (x, z),
            textcoords="offset points",
            xytext=xytext,
            fontsize=6,
            color="royalblue",
            ha="center",
            annotation_clip=False,
        )

    for x, z in sources:
        ax.scatter(x, z, marker="*", color="crimson", s=200, zorder=6, label="Source")

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower left",
        fontsize=8,
        fancybox=False,
        framealpha=1.0,
    )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_aspect("equal")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(min(bot_z), max(top_z))

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=150)
    plt.close(fig)
    print(f"Saved geometry plot to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
