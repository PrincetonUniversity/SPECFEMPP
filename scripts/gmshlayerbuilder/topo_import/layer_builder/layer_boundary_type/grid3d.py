from dataclasses import dataclass, field
from pathlib import Path
from typing import override

import numpy as np
from gmsh2meshfem.gmsh_dep import GmshContext

from ...tags import EPS
from ..layer3d import LayerBoundary3D


@dataclass
class Gridded3D(LayerBoundary3D):
    """A boundary represented on a grid"""

    suppress_utm_projection: bool
    nx: int  # number of data points (not intervals)
    ny: int  # number of data points (not intervals)
    xmin: float
    ymin: float
    x_spacing: float
    y_spacing: float
    elevations: np.ndarray = field(init=False)

    def __post_init__(self):
        self.elevations = np.empty((self.nx, self.ny))

    @property
    def xmax(self):
        return self.xmin + self.x_spacing * (self.nx - 1)

    @property
    def ymax(self):
        return self.ymin + self.y_spacing * (self.ny - 1)

    @staticmethod
    def from_file(
        suppress_utm_projection: bool,
        nxi: int,
        neta: int,
        long_min: float,
        lat_min: float,
        spacing_xi: float,
        spacing_eta: float,
        elevation_filename: Path | str,
    ):
        bd = Gridded3D(
            suppress_utm_projection=suppress_utm_projection,
            nx=nxi,
            ny=neta,
            xmin=long_min,
            ymin=lat_min,
            x_spacing=spacing_xi,
            y_spacing=spacing_eta,
        )
        elevations_flattened = np.loadtxt(Path(elevation_filename))

        # points are read in as for(y){for(x){read_value}}
        # https://github.com/SPECFEM/specfem3d/blob/master/src/meshfem3D/create_interfaces_mesh.f90#L243-L259
        # that is to say, the x-index (first one) changes faster

        bd.elevations[:, :] = np.reshape(elevations_flattened, (nxi, neta), order="F")

        return bd

    @override
    def build_layer(
        self,
        xlow: float,
        xhigh: float,
        ylow: float,
        yhigh: float,
        nonconforming_above_and_below: bool,
        gmsh: GmshContext,
    ):
        # which dims should be expanded
        pad_xlow = self.xmin - xlow > (xhigh - xlow) * EPS
        pad_xhigh = xhigh - self.xmax > (xhigh - xlow) * EPS

        pad_ylow = self.ymin - ylow > (yhigh - ylow) * EPS
        pad_yhigh = yhigh - self.ymax > (yhigh - ylow) * EPS

        def gen_axis(
            pts_interior: np.ndarray, pad_low: bool, pad_high: bool, low_val, high_val
        ):
            concat = []
            if pad_low:
                concat.append([low_val])
            concat.append(pts_interior)
            if pad_high:
                concat.append([high_val])
            return np.concatenate(concat)

        xpts = gen_axis(
            np.linspace(self.xmin, self.xmax, self.nx), pad_xlow, pad_xhigh, xlow, xhigh
        )
        ypts = gen_axis(
            np.linspace(self.ymin, self.ymax, self.ny), pad_ylow, pad_yhigh, ylow, yhigh
        )

        xgrid, ygrid = np.meshgrid(xpts, ypts, indexing="ij")
        zgrid = np.empty(xgrid.shape)

        # pad elevations array according to pad_... flags
        zgrid[
            slice(1, self.nx + 2) if pad_xlow else slice(0, self.nx + 1),
            slice(1, self.ny + 2) if pad_ylow else slice(0, self.ny + 1),
        ] = self.elevations
        if pad_xlow:
            zgrid[0, :] = zgrid[1, :]
        if pad_ylow:
            zgrid[:, 0] = zgrid[:, 1]
        if pad_xhigh:
            zgrid[-1, :] = zgrid[-2, :]
        if pad_yhigh:
            zgrid[:, -1] = zgrid[:, -2]

        # ==================================================
        # meshify
        corner_nodes = np.empty((2, 2), dtype=int)
        for ix in [0, -1]:
            for iy in [0, -1]:
                corner_nodes[ix, iy] = gmsh.model.add_discrete_entity(dim=0)
                gmsh.model.set_coordinates(
                    corner_nodes[ix, iy], xgrid[ix, iy], ygrid[ix, iy], zgrid[ix, iy]
                )
        curve_front = gmsh.model.add_discrete_entity(
            dim=1, tag=-1, boundary=[corner_nodes[0, 0], corner_nodes[1, 0]]
        )
        curve_right = gmsh.model.add_discrete_entity(
            dim=1, tag=-1, boundary=[corner_nodes[1, 0], corner_nodes[1, 1]]
        )
        curve_back = gmsh.model.add_discrete_entity(
            dim=1, tag=-1, boundary=[corner_nodes[0, 1], corner_nodes[1, 1]]
        )
        curve_left = gmsh.model.add_discrete_entity(
            dim=1, tag=-1, boundary=[corner_nodes[0, 0], corner_nodes[0, 1]]
        )

        surf = gmsh.model.add_discrete_entity(
            dim=2, tag=-1, boundary=[curve_front, curve_right, -curve_back, -curve_left]
        )

        # the local nx,ny is based on the grid, and counts the number of spacings, not the number of points
        nx, ny = zgrid.shape
        nx -= 1
        ny -= 1

        node_start = gmsh.model.mesh.get_max_node_tag() + 1
        point_inds = np.arange(node_start, node_start + (nx + 1) * (ny + 1)).reshape(
            (nx + 1, ny + 1)
        )
        coords_arr = np.stack([xgrid, ygrid, zgrid], axis=-1)

        gmsh.model.mesh.add_nodes(
            dim=2,
            tag=surf,
            nodeTags=point_inds.reshape(-1),
            coord=coords_arr.reshape(-1),
        )
        for ix in [0, -1]:
            for iy in [0, -1]:
                gmsh.model.mesh.add_elements_by_type(
                    tag=corner_nodes[ix, iy],
                    elementType=15,  # points
                    elementTags=[],
                    nodeTags=[point_inds[ix, iy]],
                )

        gmsh.model.mesh.add_elements_by_type(
            tag=curve_front,
            elementType=1,  # 2 node lines
            elementTags=[],
            nodeTags=np.stack([point_inds[:-1, 0], point_inds[1:, 0]], axis=-1).reshape(
                -1
            ),
        )
        gmsh.model.mesh.add_elements_by_type(
            tag=curve_right,
            elementType=1,  # 2 node lines
            elementTags=[],
            nodeTags=np.stack(
                [point_inds[-1, :-1], point_inds[-1, 1:]], axis=-1
            ).reshape(-1),
        )
        gmsh.model.mesh.add_elements_by_type(
            tag=curve_back,
            elementType=1,  # 2 node lines
            elementTags=[],
            nodeTags=np.stack(
                [point_inds[:-1, -1], point_inds[1:, -1]], axis=-1
            ).reshape(-1),
        )
        gmsh.model.mesh.add_elements_by_type(
            tag=curve_left,
            elementType=1,  # 2 node lines
            elementTags=[],
            nodeTags=np.stack([point_inds[0, :-1], point_inds[0, 1:]], axis=-1).reshape(
                -1
            ),
        )

        gmsh.model.mesh.add_elements_by_type(
            tag=surf,
            elementType=2,  # 3 node tri
            elementTags=[],
            nodeTags=np.stack(
                [
                    point_inds[:-1, :-1],
                    point_inds[1:, :-1],
                    point_inds[1:, 1:],
                    point_inds[:-1, :-1],
                    point_inds[1:, 1:],
                    point_inds[:-1, 1:],
                ],
                axis=-1,
            ).reshape(-1),
        )

        one_part = LayerBoundary3D.BuildResultPart(
            corner_front_left=corner_nodes[0, 0],
            corner_front_right=corner_nodes[1, 0],
            corner_back_left=corner_nodes[0, 1],
            corner_back_right=corner_nodes[1, 1],
            curve_front=curve_front,
            curve_back=curve_back,
            curve_left=curve_left,
            curve_right=curve_right,
            surface=surf,
        )

        if nonconforming_above_and_below:
            # run this same code again to generate other side
            result = self.build_layer(xlow, xhigh, ylow, yhigh, False, gmsh)
            return LayerBoundary3D.BuildResult(result.below, one_part)
        else:
            # only expect one side, so return it
            return LayerBoundary3D.BuildResult(one_part)
