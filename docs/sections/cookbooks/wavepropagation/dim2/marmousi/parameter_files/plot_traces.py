import os
import numpy as np
import obspy


def get_traces(directory):
    traces = []
    ## iterate over all seismograms
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        network, station, location, channel = filename.split(".")[:4]
        trace = np.loadtxt(f, delimiter=" ")
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


directory = "OUTPUT_FILES/results"
stream = get_traces(directory)
stream.select(component="X").plot(size=(1000, 800))
stream.select(component="Z").plot(size=(1000, 800))
