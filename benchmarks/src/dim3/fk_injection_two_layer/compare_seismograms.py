"""Compare SPECFEM++ traces with the SPECFEM3D references.

Usage: compare_seismograms.py <results> <references> <table.txt> <plot.png>

The traces to compare are those in the reference directory, named
<network>.<station>.<location>.<component>.<extension>. The plot shows one
extension per station and component: velocity where there is one, otherwise
the first extension found.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {
    "semd": "Displacement (m)",
    "semv": "Velocity (m/s)",
    "sema": "Acceleration (m/s$^2$)",
    "semp": "Pressure (Pa)",
}


def unique(values):
    """The distinct values, in order of first appearance."""
    return list(dict.fromkeys(values))


def reference_traces(references):
    """The reference traces as {(station, component): {extension: name}}."""
    traces = {}
    for path in sorted(references.glob("*.sem*")):
        _, station, _, component, extension = path.name.split(".")
        traces.setdefault((station, component), {})[extension] = path.name
    if not traces:
        raise ValueError(f"no reference traces in {references}")
    return traces


def lag_seconds(trace, reference, dt):
    """Lag of trace behind reference, refined with a parabola around the peak."""
    correlation = np.correlate(trace, reference, mode="full")
    peak = int(np.argmax(correlation))
    refinement = 0.0
    if 0 < peak < len(correlation) - 1:
        left, center, right = correlation[peak - 1 : peak + 2]
        curvature = left - 2 * center + right
        if curvature != 0:
            refinement = 0.5 * (left - right) / curvature
    return (peak - (len(reference) - 1) + refinement) * dt


def metrics(trace, reference, dt):
    reference_energy = np.sum(reference**2)
    trace_energy = np.sum(trace**2)
    return {
        "correlation": np.sum(trace * reference)
        / np.sqrt(trace_energy * reference_energy),
        "ratio": np.sqrt(trace_energy / reference_energy),
        "lag": lag_seconds(trace, reference, dt),
        "misfit": np.sqrt(np.sum((trace - reference) ** 2) / reference_energy),
    }


def check_the_comparison(reference, dt):
    """A trace against itself must be perfect, or the results mean nothing."""
    result = metrics(reference, reference, dt)
    assert np.isclose(result["correlation"], 1.0)
    assert np.isclose(result["ratio"], 1.0)
    assert abs(result["lag"]) < 1e-9
    assert result["misfit"] == 0.0


def main(results, references, table_path, plot_path):
    traces = reference_traces(references)
    stations = unique(station for station, _ in traces)
    components = unique(component for _, component in traces)
    plotted = unique(
        "semv" if "semv" in extensions else next(iter(extensions))
        for extensions in traces.values()
    )

    lines = []
    figure, axes = plt.subplots(
        len(stations),
        len(components),
        figsize=(4 * len(components), 1.6 * len(stations) + 1.5),
        sharex=True,
        squeeze=False,
    )

    lines.append(
        f"{'trace':<24}{'correlation':>13}{'ratio':>9}{'lag (s)':>10}{'misfit':>11}"
    )
    worst_station_error = 0.0
    checked = False
    for row, station in enumerate(stations):
        # As the Newmark displacement tests: one error per station, over all
        # components and types.
        station_error = 0.0
        station_norm = 0.0
        for column, component in enumerate(components):
            extensions = traces.get((station, component), {})
            for extension, name in extensions.items():
                time, reference = np.loadtxt(references / name, unpack=True)
                trace_time, trace = np.loadtxt(results / name, unpack=True)
                if not np.allclose(time, trace_time, atol=1e-3):
                    raise ValueError(f"{name}: time axes differ")
                dt = time[1] - time[0]
                if not checked:
                    check_the_comparison(trace, dt)
                    checked = True

                station_error += np.sum((trace - reference) ** 2)
                station_norm += np.sum(reference**2)

                # A silent component has nothing to correlate with.
                if np.max(np.abs(reference)) > 0:
                    result = metrics(trace, reference, dt)
                    lines.append(
                        f"{name:<24}{result['correlation']:>13.5f}"
                        f"{result['ratio']:>9.4f}{result['lag']:>10.3f}"
                        f"{result['misfit']:>11.3e}"
                    )

                if extension == (
                    "semv" if "semv" in extensions else next(iter(extensions))
                ):
                    axis = axes[row, column]
                    axis.plot(time, reference, "k", label="SPECFEM3D")
                    axis.plot(time, trace, "r--", label="SPECFEM++")
                    axis.set_title(f"{station} {component}", fontsize=9)

        error = np.sqrt(station_error / station_norm)
        worst_station_error = max(worst_station_error, error)
        lines.append(f"station {station}: Newmark-style error {error:.3e}")

    lines.append(f"worst Newmark-style error {worst_station_error:.3e}")
    axes[0, 0].legend(fontsize=8)
    figure.supxlabel("Time (s)")
    figure.supylabel(
        LABELS[plotted[0]] if len(plotted) == 1 else "Amplitude", fontsize=10
    )
    figure.tight_layout()
    figure.savefig(plot_path, dpi=150)

    table = "\n".join(lines)
    print(table)
    table_path.write_text(table + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
