# %%
# Read the sources yaml file
import matplotlib.pyplot as plt
import yaml

with open("sources.yaml") as f:
    sourcedict = yaml.safe_load(f)

source = sourcedict["sources"][0]["force"]

# %%
# Read the stations file
stations = []
networks = []
x = []
z = []
with open("OUTPUT_FILES/STATIONS") as f:
    for line in f:
        if line.startswith("#"):
            continue
        line = line.split()
        stations.append(line[0])
        networks.append(line[1])
        x.append(float(line[2]))
        z.append(float(line[3]))

# %%
# Plot the source station geometry
fig, ax = plt.subplots()
ax.scatter(x, z, c="r", label="Stations")
for ista, ix, iz in zip(stations, x, z):
    if ista in ["S0010", "S0020", "S0030", "S0040", "S0050"]:
        ax.text(ix, iz + 0.01, ista, fontsize=8)
        ax.scatter(ix, iz, c="k")
ax.scatter(source["x"], source["z"], c="b", label="Source")
ax.set_xlabel("x")
ax.set_ylabel("z")
ax.set_xlim(0, 0.33)
ax.set_ylim(0, 0.33)
ax.legend(fancybox=False, loc="lower left")
plt.savefig("geometry.png", dpi=300)
