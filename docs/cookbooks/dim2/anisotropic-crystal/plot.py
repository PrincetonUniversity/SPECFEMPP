import glob
import os
import numpy as np
import obspy
import matplotlib

matplotlib.use("agg")


def get_traces(directory):
    traces = []
    station_name = [
        "S0010",
        "S0020",
        "S0030",
        "S0040",
        "S0050",
    ]
    files = [
        glob.glob(directory + f"/{stationname}*.sem*")[0]
        for stationname in station_name
    ]

    ## iterate over all seismograms
    for filename in files:
        station_id = os.path.splitext(filename)[0]
        station_id = station_id.split("/")[-1]
        network = station_id[5:7]
        station = station_id[0:5]
        location = "00"
        component = station_id[7:10]
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
                    "channel": component,
                    "starttime": starttime,
                    "delta": dt,
                },
            )
        )

    stream = obspy.Stream(traces)

    return stream


stream = get_traces("OUTPUT_FILES/results")
stream.plot(size=(800, 1000))
