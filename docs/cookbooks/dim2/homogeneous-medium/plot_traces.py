import os
import numpy as np
import obspy


def get_traces(directory):
    traces = []
    ## iterate over all seismograms
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        station_name = os.path.splitext(filename)[0]
        trace = np.loadtxt(f, delimiter=" ")
        starttime = trace[0, 0]
        dt = trace[1, 0] - trace[0, 0]
        traces.append(
            obspy.Trace(
                trace[:, 1],
                {"network": station_name, "starttime": starttime, "delta": dt},
            )
        )

    stream = obspy.Stream(traces)

    return stream


directory = "OUTPUT_FILES/seismograms"
stream = get_traces(directory)
stream.plot(size=(800, 1000))
