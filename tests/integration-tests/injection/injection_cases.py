"""Reference traces for the end-to-end injection tests (see tests.yaml).

    python injection_cases.py analytic <case> <output dir>
        Velocity of the case's analytic plane wave at every station, sampled
        at the simulation's times t0 + s * dt.
    python injection_cases.py simulate <case> <output dir>
        Run `specfem 3d` on the case with the plane-wave plugin and keep its
        velocity seismograms (SPECFEMPP_BINDIR, PLANE_WAVE_PLUGIN).

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


def load_case(name):
    tests = yaml.safe_load((HERE / "tests.yaml").read_text())
    for case in tests["cases"]:
        if case["name"] == name:
            return tests, case
    raise SystemExit(f"injection_cases: no case '{name}' in tests.yaml")


def stations():
    """Station name -> (x, y, z); a STATIONS line is `name net y x elevation z`."""
    result = {}
    for line in (HERE / "STATIONS").read_text().splitlines():
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
    tests, case = load_case(name)
    fixture = tests["fixture"]
    config = yaml.safe_load((HERE / case["plugin-config"]).read_text())
    velocity = plane_wave_velocity(config)
    t = fixture["t0"] + fixture["dt"] * np.arange(case["nstep"])

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
    tests, case = load_case(name)
    fixture = tests["fixture"]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    mesh = HERE / tests["meshes"][case.get("mesh", "serial")]
    parameters = {
        "parameters": {
            "header": {"title": name, "description": "injection reference run"},
            "simulation-setup": {
                "quadrature": {"quadrature-type": fixture["quadrature"]},
                "solver": {
                    "time-marching": {
                        "time-scheme": {
                            "type": "Newmark",
                            "dt": fixture["dt"],
                            "nstep": case["nstep"],
                        }
                    }
                },
                "simulation-mode": {
                    "forward": {
                        "writer": {
                            "seismogram": {"format": "ascii", "directory": str(output)}
                        }
                    }
                },
            },
            "receivers": {
                "stations": str(HERE / fixture["stations"]),
                "angle": 0.0,
                "seismogram-type": ["velocity"],
                "nstep_between_samples": 1,
            },
            "run-setup": {"number-of-processors": 1, "number-of-runs": 1},
            "databases": {"mesh-database": str(mesh)},
            "sources": str(HERE / case["source"]),
            "injection": {"method": "stacey"},
        }
    }
    parameter_file = output / "parameters.yaml"
    parameter_file.write_text(yaml.safe_dump(parameters))
    subprocess.run(
        [
            f"{os.environ['SPECFEMPP_BINDIR']}/specfem",
            "3d",
            "-p",
            str(parameter_file),
            "--injection-plugin",
            os.environ["PLANE_WAVE_PLUGIN"],
            "--injection-config",
            str(HERE / case["plugin-config"]),
        ],
        check=True,
    )
    parameter_file.unlink()
    files = sorted(p.name for p in output.glob("*.semv"))
    (output / "trace_list.txt").write_text("\n".join(files) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] not in ("analytic", "simulate"):
        raise SystemExit(__doc__)
    {"analytic": write_analytic, "simulate": simulate}[sys.argv[1]](*sys.argv[2:])
