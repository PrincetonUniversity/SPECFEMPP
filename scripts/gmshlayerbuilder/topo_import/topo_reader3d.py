import shlex
from pathlib import Path
from typing import Any

import numpy as np

from .layer_builder.layer3d import Layer3D
from .layer_builder.layer_boundary_type.grid3d import Gridded3D
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
    depth_block_m: float | None = None,
) -> LayeredBuilder3D:
    with Path(file).open("r") as f:
        parser = shlex.shlex(f)
        parser.whitespace_split = True

        token: Any = parser.get_token()
        ninterfaces: int = 0

        try:
            ninterfaces = int(token)
        except (TypeError, ValueError) as cause:
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

            token = parser.get_token()

            if token.lower() == ".true.":
                suppress_utm_projection = True
            elif token.lower() == ".false.":
                suppress_utm_projection = False
            else:
                e = RuntimeError(
                    f"Unable to parse SUPPRESS_UTM_PROJECTION of interface {iinterface + 1}: {token}"
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
            except (TypeError, ValueError) as cause:
                e = RuntimeError(
                    f"Unable to parse nx,ny of interface {iinterface + 1}."
                )
                raise e from cause

            x_token: Any = parser.get_token()
            y_token: Any = parser.get_token()
            try:
                xmin_layer = float(x_token)
                ymin_layer = float(y_token)
            except (TypeError, ValueError) as cause:
                e = RuntimeError(
                    f"Unable to parse LONG_MIN,LAT_MIN (layer xmin, layer ymin) of interface {iinterface + 1}."
                )
                raise e from cause

            x_token: Any = parser.get_token()
            y_token: Any = parser.get_token()
            try:
                xinc = float(x_token)
                yinc = float(y_token)
            except (TypeError, ValueError) as cause:
                e = RuntimeError(
                    f"Unable to parse LONG_MIN,LAT_MIN (layer xmin, layer ymin) of interface {iinterface + 1}."
                )
                raise e from cause

            elevation_filename = parser.get_token()
            if elevation_filename is None:
                e = RuntimeError(
                    f"Unable to load elevation file of interface {iinterface + 1}. Must not be `None`"
                )
                raise e

            elevation_path = Path(file).parent / Path(elevation_filename)
            if not elevation_path.is_file():
                e = RuntimeError(
                    f"Unable to load elevation file of interface {iinterface + 1}: {elevation_filename}"
                )
                e.add_note(f"Resolved to {str(elevation_path.resolve())}")
                raise e

            if not suppress_utm_projection:
                e = NotImplementedError(
                    "SUPPRESS_UTM_PROJECTION == .false. is not supported. Please use Cartesian coordinates."
                )
                raise e

            bd = Gridded3D.from_file(
                suppress_utm_projection=suppress_utm_projection,
                nxi=nx,
                neta=ny,
                long_min=xmin_layer,
                lat_min=ymin_layer,
                spacing_xi=xinc,
                spacing_eta=yinc,
                elevation_filename=elevation_path,
            )
            xmin = min(xmin, bd.xmin)
            xmax = max(xmax, bd.xmax)
            ymin = min(ymin, bd.ymin)
            ymax = max(ymax, bd.ymax)
            layer_boundaries.append(bd)

        # prepend bottom floor if set
        if depth_block_m is not None:
            ninterfaces += 1
            bd = Gridded3D(
                suppress_utm_projection=True,
                nx=2,
                ny=2,
                xmin=xmin,
                ymin=ymin,
                x_spacing=xmax - xmin,
                y_spacing=ymax - ymin,
            )
            bd.elevations[...] = depth_block_m
            layer_boundaries.insert(0, bd)

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
            z_token: Any = parser.get_token()
            try:
                nz = int(z_token)
            except (TypeError, ValueError) as cause:
                e = RuntimeError(f"Unable to parse nz of layer {ilayer + 1}.")
                raise e from cause

            # guess nx by attempting to have aspect ratio 1. We need height of layer
            zavg_below = np.mean(layer_boundaries[ilayer].elevations)
            zavg_above = np.mean(layer_boundaries[ilayer + 1].elevations)
            layer_aspect_ratio_x = (xmax - xmin) / (zavg_above - zavg_below)
            layer_aspect_ratio_y = (ymax - ymin) / (zavg_above - zavg_below)
            nx = max(1, round(nz * layer_aspect_ratio_x))
            ny = max(1, round(nz * layer_aspect_ratio_y))
            if materialtype_strcode is None:
                layers.append(Layer3D(nx, ny, nz))
            else:
                layers.append(
                    Layer3D(
                        nx,
                        ny,
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
