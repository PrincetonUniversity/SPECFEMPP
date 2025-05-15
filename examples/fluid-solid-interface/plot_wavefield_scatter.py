# %%

import numpy as np
import matplotlib.pyplot as plt
import h5py

with h5py.File("OUTPUT_FILES/wavefield/ForwardWavefield.h5", "r") as f:
    acoustic_x = f["/Coordinates/acoustic/X"][:].flatten()
    acoustic_z = f["/Coordinates/acoustic/Z"][:].flatten()
    elastic_x = f["/Coordinates/elastic_psv/X"][:].flatten()
    elastic_z = f["/Coordinates/elastic_psv/Z"][:].flatten()

    # Get steps from list of keys
    keys = list(f["/"].keys())
    keys.remove("Coordinates")
    steps = [int(x[4:]) for x in list(keys)]

    steps.sort()
    it = steps[0]
    ft = steps[-1]
    delta_timestep = steps[1] - steps[0]

    stepstr = f"Step{it + delta_timestep:06d}"

    # Get acoustic wavefield
    u_acoustic = f[f"/{stepstr}/acoustic/Potential"][:].flatten()

    # Get elastic wavefield
    u_elastic = f[f"/{stepstr}/elastic_psv/Displacement"][:]
    u_elastic = u_elastic.flatten().reshape(u_elastic.shape[1], u_elastic.shape[0]).T
    u_tot = np.sqrt(u_elastic[:, 0] ** 2 + u_elastic[:, 1] ** 2)


plt.figure(figsize=(10, 8))
plt.xlabel("X (m)")
plt.ylabel("Z (m)")
# plt.plot(acoustic_x, acoustic_z, "o", markersize=1, color="blue", label="Acoustic")
# plt.plot(elastic_x, elastic_z, "o", markersize=1, color="red", label="Elastic")
plt.scatter(acoustic_x, acoustic_z, c=np.abs(u_acoustic), s=1, cmap="Blues")
plt.scatter(elastic_x, elastic_z, c=u_tot, s=1, cmap="Grays")
# plt.xlim([acoustic_x.min(), acoustic_x.max()])
# plt.ylim([acoustic_z.min(), acoustic_z.max()])

plt.gca().set_aspect("equal", adjustable="box")
# plt.grid()
# %%
