import numpy as np

dx = 1.0
dz = 1.0

nx = int(400 / dx) + 1
nz = int(400 / dz) + 1

n = nx * nz

# model = {
#     'x': np.zeros(n),
#     'z': np.zeros(n),
#     'rho': np.ones(n) * 2700.0,
#     'vp': np.ones(n) * 3000.0,
#     'vs': np.ones(n) * 1732.051
# }

model = {
    "x": np.zeros(n),
    "z": np.zeros(n),
    "rho": np.ones(n) * 2700.0,
    "lambda": np.ones(n) * 8.1e9,
    "mu": np.ones(n) * 8.1e9,
    "nu": np.ones(n) * 2.7e5,
    "j": np.ones(n) * 2700.0,
    "lambda_c": np.ones(n) * 7.75e11,
    "mu_c": np.ones(n) * 1.5e11,
    "nu_c": np.ones(n) * 2.7e5,
}

for i in range(nx):
    for j in range(nz):
        model["x"][i * nz + j] = i * dx
        model["z"][i * nz + j] = j * dz

npt = np.array([n], dtype="int32")

for m in model:
    with open(f"./examples/spin2/model/proc000000_{m}.bin", "w") as f:
        f.seek(0)
        npt.tofile(f)
        f.seek(4)
        model[m].astype("float32").tofile(f)
