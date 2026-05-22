import glob
import os
import numpy as np
import obspy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def plot_comparison(stream, stream_att, component, outfile):
    sel = stream.select(component=component)
    sel_att = stream_att.select(component=component)
    stations = sorted(set(tr.stats.station for tr in sel))
    fig, axes = plt.subplots(
        len(stations), 1, figsize=(10, 2 * len(stations)), sharex=True
    )
    if len(stations) == 1:
        axes = [axes]
    for ax, station in zip(axes, stations):
        tr = sel.select(station=station)[0]
        tr_att = sel_att.select(station=station)[0]
        ax.plot(tr.times(), tr.data, label="attenuation off", color="black")
        ax.plot(
            tr_att.times(),
            tr_att.data,
            label="attenuation on",
            color="red",
            linestyle="--",
        )
        ax.set_ylabel(station)
        ax.legend(loc="upper right", fontsize=6)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Component {component}: attenuation on vs. off")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


stream = get_traces("OUTPUT_FILES/attenuation_off/results")
stream_att = get_traces("OUTPUT_FILES/attenuation_on/results")
os.makedirs("OUTPUT_FILES/results", exist_ok=True)
plot_comparison(stream, stream_att, "X", "OUTPUT_FILES/results/traces_X.png")
plot_comparison(stream, stream_att, "Z", "OUTPUT_FILES/results/traces_Z.png")
