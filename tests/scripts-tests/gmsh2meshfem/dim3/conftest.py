import numpy as np
import pytest
from gmsh2meshfem.gmsh_dep import GmshContext


@pytest.fixture
def box_builder():
    def execute(
        gmsh: GmshContext,
        nx: int,
        ny: int,
        nz: int,
        bbox_min: tuple[float, float, float] = (-1, -1, -1),
        bbox_max: tuple[float, float, float] = (1, 1, 1),
    ):
        x_vals = [bbox_min[0], bbox_max[0]]
        y_vals = [bbox_min[1], bbox_max[1]]
        z_vals = [bbox_min[2], bbox_max[2]]

        corners = np.empty((2, 2, 2), dtype=int)
        for ix in range(2):
            for iy in range(2):
                for iz in range(2):
                    corners[ix, iy, iz] = gmsh.model.geo.add_point(
                        x_vals[ix], y_vals[iy], z_vals[iz]
                    )

        lines_xyz = np.empty((3, 2, 2), dtype=int)
        for i in range(2):
            for j in range(2):
                # x
                lines_xyz[0, i, j] = gmsh.model.geo.add_line(
                    corners[0, i, j], corners[1, i, j]
                )
                # y
                lines_xyz[1, i, j] = gmsh.model.geo.add_line(
                    corners[i, 0, j], corners[i, 1, j]
                )
                # z
                lines_xyz[2, i, j] = gmsh.model.geo.add_line(
                    corners[i, j, 0], corners[i, j, 1]
                )

        surfaces = [
            gmsh.model.geo.add_plane_surface([gmsh.model.geo.add_curve_loop(lines)])
            for lines in [
                [  # left (ix = 0)
                    lines_xyz[1, 0, 0],
                    lines_xyz[2, 0, 1],
                    -lines_xyz[1, 0, 1],
                    -lines_xyz[2, 0, 0],
                ],
                [  # right (ix = 1)
                    lines_xyz[1, 1, 0],
                    lines_xyz[2, 1, 1],
                    -lines_xyz[1, 1, 1],
                    -lines_xyz[2, 1, 0],
                ],
                [  # front (iy = 0)
                    lines_xyz[0, 0, 0],
                    lines_xyz[2, 1, 0],
                    -lines_xyz[0, 0, 1],
                    -lines_xyz[2, 0, 0],
                ],
                [  # back (iy = 1)
                    lines_xyz[0, 1, 0],
                    lines_xyz[2, 1, 1],
                    -lines_xyz[0, 1, 1],
                    -lines_xyz[2, 0, 1],
                ],
                [  # bottom (iz = 0)
                    lines_xyz[0, 0, 0],
                    lines_xyz[1, 1, 0],
                    -lines_xyz[0, 1, 0],
                    -lines_xyz[1, 0, 0],
                ],
                [  # top (iz = 1)
                    lines_xyz[0, 0, 1],
                    lines_xyz[1, 1, 1],
                    -lines_xyz[0, 1, 1],
                    -lines_xyz[1, 0, 1],
                ],
            ]
        ]

        volume = gmsh.model.geo.add_volume([gmsh.model.geo.add_surface_loop(surfaces)])

        yield volume

        subdiv_per_dim = [nx + 1, ny + 1, nz + 1]

        for idim in range(3):
            for i in range(2):
                for j in range(2):
                    gmsh.model.mesh.set_transfinite_curve(
                        lines_xyz[idim, i, j], subdiv_per_dim[idim]
                    )

        for surf in surfaces:
            gmsh.model.mesh.set_transfinite_surface(surf)
            gmsh.model.mesh.set_recombine(2, surf)  # quads

        gmsh.model.mesh.set_transfinite_volume(volume)
        return

    class BoxBuilder:
        def __init__(self):
            self.held = []

        def build(
            self,
            gmsh: GmshContext,
            nx: int,
            ny: int,
            nz: int,
            bbox_min: tuple[float, float, float] = (-1, -1, -1),
            bbox_max: tuple[float, float, float] = (1, 1, 1),
        ):
            executor = execute(
                gmsh=gmsh, nx=nx, ny=ny, nz=nz, bbox_min=bbox_min, bbox_max=bbox_max
            )
            self.held.append(executor)
            return next(executor)

        def post_geo_sync(self):
            for executor in self.held:
                # finalize
                for _ in executor:
                    ...
            self.held = []

    return BoxBuilder()


@pytest.fixture(params=[])
def simple_volume(request): ...
