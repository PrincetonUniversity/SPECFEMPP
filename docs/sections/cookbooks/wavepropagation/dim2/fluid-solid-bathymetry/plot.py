import os
import numpy as np
import obspy


def get_traces(directory):
    traces = []

    # Iterate over all pressure seismogram files
    for filename in os.listdir(directory):
        if filename.endswith(".semp"):
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


# Load pressure seismograms from acoustic domain
directory = "OUTPUT_FILES/seismograms"
stream = get_traces(directory)

# Plot pressure recordings (acoustic domain)
# Note: Since receivers are in water, we expect pressure recordings
stream.plot(size=(1000, 800))
