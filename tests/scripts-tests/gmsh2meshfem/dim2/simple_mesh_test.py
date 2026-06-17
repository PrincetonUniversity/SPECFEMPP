import numpy as np

from gmsh2meshfem.gmsh_dep import GmshContext
from gmsh2meshfem.dim2.model import Model


def test_three_squares_single_surface():
    with GmshContext() as gmsh:
        square_size = 1.0
        nx = 1
        nz = 3
        p1 = gmsh.model.geo.add_point(0, 0, 0)
        p2 = gmsh.model.geo.add_point(square_size * nx, 0, 0)
        p3 = gmsh.model.geo.add_point(square_size * nx, 0, square_size * nz)
        p4 = gmsh.model.geo.add_point(0, 0, square_size * nz)

        line_bottom = gmsh.model.geo.add_line(p1, p2)
        line_right = gmsh.model.geo.add_line(p2, p3)
        line_top = gmsh.model.geo.add_line(p3, p4)
        line_left = gmsh.model.geo.add_line(p4, p1)

        gmsh.model.geo.mesh.set_transfinite_curve(line_bottom, nx + 1)
        gmsh.model.geo.mesh.set_transfinite_curve(line_top, nx + 1)
        gmsh.model.geo.mesh.set_transfinite_curve(line_left, nz + 1)
        gmsh.model.geo.mesh.set_transfinite_curve(line_right, nz + 1)

        loop = gmsh.model.geo.add_curve_loop(
            [line_bottom, line_right, line_top, line_left]
        )

        surface = gmsh.model.geo.add_plane_surface([loop])

        # ngnod = 9 quads
        gmsh.model.geo.mesh.set_transfinite_surface(surface)
        gmsh.model.geo.mesh.set_recombine(2, surface)  # quads
        gmsh.option.setNumber("Mesh.ElementOrder", 2)

        gmsh.model.geo.synchronize()

        gmsh.model.mesh.generate()

        model = Model.from_meshed_surface(surface, gmsh=gmsh)

        assert model.nodes.shape == ((nx * 2 + 1) * (nz * 2 + 1), 3), (
            f"An order-2 grid of {nx} x {nz} elements expects {nx * 2 + 1} nodes along x "
            f"and {nz * 2 + 1} nodes along y."
        )

        nelem = nx * nz
        assert model.elements.shape == (nelem, 9), (
            f"An order-2 grid of {nx} x {nz} elements expects {nelem} elements."
        )

        np.testing.assert_array_equal(
            model.materials,
            np.array([1] * nelem),
            f"All {nelem} elements should be material index 1.",
        )
