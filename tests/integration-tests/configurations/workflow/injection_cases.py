"""Reference traces for the injection test cases (configurations/dim3/injection).

    python injection_cases.py analytic <case> <output dir>
        Velocity of the case's analytic plane wave at every station, sampled
        at the simulation's times t0 + s * dt.
    python injection_cases.py simulate <case> <output dir>
        Run `specfem 3d` on the case with the plane-wave plugin and keep its
        velocity seismograms (SPECFEMPP_BINDIR, PLANE_WAVE_PLUGIN).

A case is a directory of configurations/dim3/injection holding a config.yaml and
a specfem_config.yaml; the time step, the number of steps and the plugin
configuration are read from them.

The plane wave is written out here independently of the plugin's C++, so the
tests also check that both follow the same conventions:
k = (sin t cos a, sin t sin a, cos t) for take-off t from +z and azimuth a from
+x; d = k for P and dk/dt for SV; v = A d f'(tau),
tau = t - t_origin - k.(x - x0)/c, with f the displacement waveform.
"""

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
INJECTION = HERE.parent / "dim3" / "injection"
# The paths in a specfem_config.yaml are relative to the tests' working directory
# (tests/integration-tests), where `configurations/` is found.
TESTS_ROOT = HERE.parent.parent


def load_case(name):
    """The case directory, its config.yaml and the `parameters` of its specfem_config.yaml."""
    directory = INJECTION / name
    if not (directory / "config.yaml").is_file():
        raise SystemExit(f"injection_cases: no case '{name}' in {INJECTION}")
    config = yaml.safe_load((directory / "config.yaml").read_text())
    parameters = yaml.safe_load((directory / "specfem_config.yaml").read_text())[
        "parameters"
    ]
    return directory, config, parameters


def time_scheme(parameters):
    return parameters["simulation-setup"]["solver"]["time-marching"]["time-scheme"]


def stations():
    """Station name -> (x, y, z); a STATIONS line is `name net y x elevation z`."""
    result = {}
    for line in (INJECTION / "plugin" / "STATIONS").read_text().splitlines():
        name, network, y, x, _, z = line.split()
        result[f"{network}.{name}"] = np.array([float(x), float(y), float(z)])
    return result


def waveform_derivative(stf):
    """f'(tau) of the displacement waveform."""
    if stf["type"] == "ricker":
        a = (math.pi * stf["f0"]) ** 2
        return lambda tau: 2 * a * tau * (2 * a * tau**2 - 3) * np.exp(-a * tau**2)
    a = 1 / stf["half-duration"] ** 2
    return lambda tau: -2 * a * tau * np.exp(-a * tau**2)


def plane_wave_velocity(config):
    """Function (x, t) -> velocity [len(t), 3] of the configured plane wave."""
    material, wave = config["material"], config["wave"]
    theta, phi = math.radians(wave["take-off"]), math.radians(wave["azimuth"])
    k = np.array(
        [
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta),
        ]
    )
    if wave["type"] == "P":
        d, c = k, material["vp"]
    else:
        d = np.array(
            [
                math.cos(theta) * math.cos(phi),
                math.cos(theta) * math.sin(phi),
                -math.sin(theta),
            ]
        )
        c = material["vs"]
    fp = waveform_derivative(config["source-time-function"])
    origin = np.array(wave["origin"], dtype=float)
    amplitude, origin_time = wave["amplitude"], wave["origin-time"]

    def velocity(x, t):
        tau = t - origin_time - k.dot(x - origin) / c
        return amplitude * np.outer(fp(tau), d)

    return velocity


def write_analytic(name, output):
    directory, case, parameters = load_case(name)
    scheme = time_scheme(parameters)
    plugin = yaml.safe_load(
        (directory / case["injection"]["plugin-config"]).read_text()
    )
    velocity = plane_wave_velocity(plugin)
    t = case["injection"]["t0"] + scheme["dt"] * np.arange(scheme["nstep"])

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for station, x in stations().items():
        v = velocity(x, t)
        for component, axis in zip("XYZ", range(3)):
            path = output / f"{station}.S3.HX{component}.semv"
            np.savetxt(path, np.column_stack([t, v[:, axis]]), fmt="%.9e")
            files.append(path.name)
    (output / "trace_list.txt").write_text("\n".join(files) + "\n")


def simulate(name, output):
    directory, case, parameters = load_case(name)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    parameters["simulation-setup"]["simulation-mode"]["forward"]["writer"][
        "seismogram"
    ]["directory"] = str(output)
    parameter_file = output / "parameters.yaml"
    parameter_file.write_text(yaml.safe_dump({"parameters": parameters}))
    subprocess.run(
        [
            f"{os.environ['SPECFEMPP_BINDIR']}/specfem",
            "3d",
            "-p",
            str(parameter_file),
            "--injection-plugin",
            os.environ["PLANE_WAVE_PLUGIN"],
            "--injection-config",
            str((directory / case["injection"]["plugin-config"]).resolve()),
        ],
        cwd=TESTS_ROOT,
        check=True,
    )
    parameter_file.unlink()
    files = sorted(p.name for p in output.glob("*.semv"))
    (output / "trace_list.txt").write_text("\n".join(files) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] not in ("analytic", "simulate"):
        raise SystemExit(__doc__)
    {"analytic": write_analytic, "simulate": simulate}[sys.argv[1]](*sys.argv[2:])
