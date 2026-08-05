from pathlib import Path

from .layer_builder.layer2d import Layer2D
from .layer_builder.layer_boundary_type.lerp import LerpLayerBoundary2D
from .layer_builder.layeredbuilder2d import LayeredBuilder2D
from .tags import IS_FLUID_PER_MATERIAL_STRCODE, BoundaryConditionType


# processes out comments from topo file
def _file_get_line(file_input_stream):
    line = None

    # we may read a bunch of blank lines (or commented)
    while not line:
        line = file_input_stream.readline()
        # exit criterion: line is already blank (no newline)
        if not line:
            return line

        # clear whitespace
        line = line.replace("\n", "").strip()

        if "#" in line:
            line = line.split("#")[0].strip()

    return line


def builder_from_topo_file2d(
    file: Path | str,
    set_left_boundary: BoundaryConditionType = "neumann",
    set_right_boundary: BoundaryConditionType = "neumann",
    set_top_boundary: BoundaryConditionType = "neumann",
    set_bottom_boundary: BoundaryConditionType = "neumann",
    materialtype_strcode: str | None = None,
) -> LayeredBuilder2D:
    with Path(file).open("r") as f:
        ninterfaces = int(_file_get_line(f))
        layer_boundaries = []

        # keep a tally on smallest and largest x values
        xmin = float("inf")
        xmax = float("-inf")

        for iinterface in range(ninterfaces):
            # each interface gives a set of points. Read those into bd:
            bd = LerpLayerBoundary2D()

            npoints = int(_file_get_line(f))
            for ipoint in range(npoints):
                read_in = _file_get_line(f).split()
                try:
                    assert len(read_in) == 2
                    x = float(read_in[0])
                    y = float(read_in[1])
                    bd.points.append((x, y))
                    xmin = min(x, xmin)
                    xmax = max(x, xmax)
                except (ValueError, AssertionError) as e:
                    msg = (
                        f'Failed to parse topography file "{str(file)}".'
                        f'Cannot recover 2D point from "{str(read_in)}"'
                    )
                    raise RuntimeError(msg) from e
            layer_boundaries.append(bd)

        # points complete, recover num cells in vertical for each layer
        nlayers = ninterfaces - 1
        if materialtype_strcode is not None and len(materialtype_strcode) != nlayers:
            e = ValueError(
                f"Material type string code '{materialtype_strcode}' is of length "
                f"{len(materialtype_strcode)}, but the topography file specifies {nlayers} layers!"
            )
            e.add_note("Make sure that there is only one material type per layer.")
            raise e

        layers = []
        for ilayer in range(nlayers):
            nz = int(_file_get_line(f))

            # guess nx by attempting to have aspect ratio 1. We need height of layer
            zavg_below = sum(z for x, z in layer_boundaries[ilayer].points) / len(
                layer_boundaries[ilayer].points
            )
            zavg_above = sum(z for x, z in layer_boundaries[ilayer + 1].points) / len(
                layer_boundaries[ilayer + 1].points
            )
            layer_aspect_ratio = (xmax - xmin) / (zavg_above - zavg_below)
            nx = max(1, round(nz * layer_aspect_ratio))
            if materialtype_strcode is None:
                layers.append(Layer2D(nx, nz))
            else:
                layers.append(
                    Layer2D(
                        nx,
                        nz,
                        skip_acoustic_free_surface=not IS_FLUID_PER_MATERIAL_STRCODE[
                            materialtype_strcode[ilayer]
                        ],
                    )
                )

        builder = LayeredBuilder2D(
            xmin,
            xmax,
            set_bottom_boundary=set_bottom_boundary,
            set_top_boundary=set_top_boundary,
            set_left_boundary=set_left_boundary,
            set_right_boundary=set_right_boundary,
        )
        builder.layers = layers
        builder.boundaries = layer_boundaries

        return builder
