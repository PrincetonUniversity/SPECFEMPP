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
        network, station, location, channel = station_name.split(".")[:4]
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


stream = get_traces("OUTPUT_FILES/seismograms")
stream.plot(size=(1000, 750)).savefig("OUTPUT_FILES/seismograms.png", dpi=300)
