import shlex
from pathlib import Path
from typing import Any

from .layer_builder.layeredbuilder3d import LayeredBuilder3D
from .tags import IS_FLUID_PER_MATERIAL_STRCODE, BoundaryConditionType


def builder_from_topo_file3d(
    file: Path | str,
    set_left_boundary: BoundaryConditionType = "neumann",
    set_right_boundary: BoundaryConditionType = "neumann",
    set_top_boundary: BoundaryConditionType = "neumann",
    set_bottom_boundary: BoundaryConditionType = "neumann",
    set_front_boundary: BoundaryConditionType = "neumann",
    set_back_boundary: BoundaryConditionType = "neumann",
    materialtype_strcode: str | None = None,
) -> LayeredBuilder3D:
    with Path(file).open("r") as f:
        parser = shlex.shlex(f)

        token: Any = parser.get_token()
        ninterfaces: int = 0

        try:
            ninterfaces = int(token)
        except TypeError | ValueError as cause:
            e = RuntimeError("Unable to parse number of interfaces.")
            e.add_note("The first line must give the number of interfaces.")
            raise e from cause
        layer_boundaries = []

        # keep a tally on smallest and largest x values
        xmin = float("inf")
        xmax = float("-inf")
        ymin = float("inf")
        ymax = float("-inf")

        for iinterface in range(ninterfaces):
            # each interface gives a set of points. Read those into bd:
            bd = LerpLayerBoundary()

            token = parser.get_token()

            if token.lower() == ".true.":
                suppress_utm_projection = True
            elif token.lower() == ".false.":
                suppress_utm_projection = False
            else:
                e = RuntimeError(
                    f"Unable to parse SUPPRESS_UTM_PROJECTION of interface {iinterface}: {token}"
                )
                raise e

            nx: int = 0
            ny: int = 0
            xmin_layer: float = 0
            ymin_layer: float = 0
            xinc: float = 0
            yinc: float = 0

            x_token: Any = parser.get_token()
            y_token: Any = parser.get_token()
            try:
                nx = int(x_token)
                ny = int(y_token)
            except TypeError | ValueError as cause:
                e = RuntimeError(f"Unable to parse nx,ny of interface {iinterface}.")
                raise e from cause

            x_token: Any = parser.get_token()
            y_token: Any = parser.get_token()
            try:
                xmin_layer = float(x_token)
                ymin_layer = float(y_token)
            except TypeError | ValueError as cause:
                e = RuntimeError(
                    f"Unable to parse LONG_MIN,LAT_MIN (layer xmin, layer ymin) of interface {iinterface}."
                )
                raise e from cause

            x_token: Any = parser.get_token()
            y_token: Any = parser.get_token()
            try:
                xinc = float(x_token)
                yinc = float(y_token)
            except TypeError | ValueError as cause:
                e = RuntimeError(
                    f"Unable to parse LONG_MIN,LAT_MIN (layer xmin, layer ymin) of interface {iinterface}."
                )
                raise e from cause

            token = parser.get_token()

            if token is None or not Path(token).is_file():
                e = RuntimeError(
                    f"Unable to load elevation file of interface {iinterface}: {token}"
                )
                raise e

            if not suppress_utm_projection:
                e = NotImplementedError(
                    "SUPPRESS_UTM_PROJECTION == .false. is not supported. Please use Cartesian coordinates."
                )
                raise e

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
                layers.append(Layer(nx, nz))
            else:
                layers.append(
                    Layer(
                        nx,
                        nz,
                        skip_acoustic_free_surface=not IS_FLUID_PER_MATERIAL_STRCODE[
                            materialtype_strcode[ilayer]
                        ],
                    )
                )

        builder = LayeredBuilder3D(
            xmin,
            xmax,
            ymin,
            ymax,
            set_bottom_boundary=set_bottom_boundary,
            set_top_boundary=set_top_boundary,
            set_left_boundary=set_left_boundary,
            set_right_boundary=set_right_boundary,
            set_front_boundary=set_front_boundary,
            set_back_boundary=set_back_boundary,
        )
        builder.layers = layers
        builder.boundaries = layer_boundaries

        return builder
