import functools
from typing import override

from .layer import LayerBoundary, EPS
from _gmsh2meshfem.gmsh_dep import GmshContext


class LerpLayerBoundary(LayerBoundary):
    """A boundary represented using a piecewise-linear curve."""

    points: list[tuple[float, float]]

    def __init__(self):
        self.points = []

    @override
    def build_layer(self, xlow: float, xhigh: float, gmsh: GmshContext):
        left_to_right = sorted(
            self.points, key=functools.cmp_to_key(lambda a, b: a[0] > b[0])
        )

        # if first and last points do not extend to the end, add horizontal to end
        if left_to_right[0][0] - xlow > (xhigh - xlow) * EPS:
            left_to_right = [(xlow, left_to_right[0][1]), *left_to_right]
        if xhigh - left_to_right[-1][0] > (xhigh - xlow) * EPS:
            left_to_right = [*left_to_right, (xhigh, left_to_right[-1][1])]
        verts = [gmsh.model.geo.add_point(x, 0, z) for x, z in left_to_right]
        return LayerBoundary.BuildResult(
            left_vertex=verts[0],
            right_vertex=verts[-1],
            curve=gmsh.model.geo.add_polyline(verts),
        )
