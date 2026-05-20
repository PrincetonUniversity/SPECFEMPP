"""Plot reference-frequency sweep: rainbow colormap, center frequency in black."""

import glob
import os
import numpy as np
import obspy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

F0 = 30.0  # Hz — reference frequency (center of sweep)
N_FREQS = 9  # 4 below + center + 4 above

REFERENCE_FREQS = np.logspace(np.log10(F0 / 10), np.log10(F0 * 10), N_FREQS)
FREQ_TAGS = [f"{i:02d}" for i in range(N_FREQS)]
CENTER_IDX = int(np.argmin(np.abs(REFERENCE_FREQS - F0)))

cmap = matplotlib.colormaps["rainbow"]
COLORS = [cmap(i / (N_FREQS - 1)) for i in range(N_FREQS)]
COLORS[CENTER_IDX] = "black"


def get_traces(directory):
    traces = []
    files = glob.glob(directory + "/*.sem*")
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
    return obspy.Stream(traces)


def plot_sweep(streams, component, outfile):
    all_stations = set()
    for st in streams:
        for tr in st.select(component=component):
            all_stations.add(tr.stats.station)
    stations = sorted(all_stations)

    fig, axes = plt.subplots(
        len(stations), 1, figsize=(10, 2 * len(stations)), sharex=True
    )
    if len(stations) == 1:
        axes = [axes]

    for ax, station in zip(axes, stations):
        plot_order = [i for i in range(N_FREQS) if i != CENTER_IDX] + [CENTER_IDX]
        for i in plot_order:
            sel = streams[i].select(component=component, station=station)
            if not sel:
                continue
            tr = sel[0]
            lw = 2.0 if i == CENTER_IDX else 0.8
            lbl = (
                f"{REFERENCE_FREQS[i]:.3g} Hz (f\u2080)"
                if i == CENTER_IDX
                else f"{REFERENCE_FREQS[i]:.3g} Hz"
            )
            ax.plot(tr.times(), tr.data, color=COLORS[i], linewidth=lw, label=lbl)
        ax.set_ylabel(station)

    axes[-1].set_xlabel("Time (s)")

    handles, labels = axes[0].get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda k: float(labels[k].split()[0]))
    fig.legend(
        [handles[k] for k in order],
        [labels[k] for k in order],
        loc="upper right",
        fontsize=6,
        ncol=2,
        fancybox=False,
        framealpha=1.0,
    )
    fig.suptitle(f"Component {component}: reference-frequency sweep")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"Saved {outfile}")


streams = [get_traces(f"OUTPUT_FILES/freq_{tag}/results") for tag in FREQ_TAGS]
os.makedirs("OUTPUT_FILES/results", exist_ok=True)
plot_sweep(streams, "X", "OUTPUT_FILES/results/traces_X.png")
plot_sweep(streams, "Z", "OUTPUT_FILES/results/traces_Z.png")
