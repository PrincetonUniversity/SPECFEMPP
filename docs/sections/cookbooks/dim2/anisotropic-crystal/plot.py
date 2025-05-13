import glob
import os
import numpy as np
import obspy


def get_traces(directory):
    traces = []
    station_name = [
        "S0010",
        "S0020",
        "S0030",
        "S0040",
        "S0050",
    ]
    files = []
    for station in station_name:
        files += glob.glob(directory + f"/??.{station}.S2.BX?.semd")

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


stream = get_traces("OUTPUT_FILES/results")
stream.select(component="X").plot(size=(1000, 750))
stream.select(component="Z").plot(size=(1000, 750))
