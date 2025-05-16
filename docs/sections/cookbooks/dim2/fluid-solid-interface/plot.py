import glob
import os
import numpy as np
import obspy


def get_traces(directory):
    traces = []

    files = glob.glob(directory + "/AA.S00??.S2.BX?.semd")
    files.sort()

    ## iterate over all seismograms
    for filename in files:
        station_id = os.path.splitext(filename)[0]
        station_id = station_id.split("/")[-1]
        network, station, location, channel = station_id.split(".")[:4]
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
stream.select(component="X").plot(size=(1000, 750))
stream.select(component="Z").plot(size=(1000, 750))
