import glob
import os
import numpy as np
import obspy

# Set matplotlib gui off
import matplotlib

matplotlib.use("Agg")


def get_traces(directory):
    traces = []
    files = glob.glob(directory + "/*.sem*")
    ## iterate over all seismograms
    for filename in files:
        station_name = os.path.splitext(filename)[0]
        network, station, location, channel = station_name.split("/")[-1].split(".")
        trace = np.loadtxt(filename)
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


stream = get_traces("OUTPUT_FILES/results")
stream.select(component="X").plot(size=(800, 1000)).savefig(
    "OUTPUT_FILES/results/traces_X.png"
)
stream.select(component="Z").plot(size=(800, 1000)).savefig(
    "OUTPUT_FILES/results/traces_Z.png"
)
