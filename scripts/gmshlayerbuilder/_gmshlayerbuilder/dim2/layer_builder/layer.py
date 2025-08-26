from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from _gmshlayerbuilder.gmsh_dep import GmshContext


@dataclass
class Layer:
    """A "layer" denoes a region spanning the width of the domain between two `LayerBoundary`s.
    Each Layer is meshed as a deformed grid (gmsh-transfinite) with specified number of cells
    along the horizontal axis (nx) and vertical axis (nz).
    """
    @dataclass
    class BuildResult:
        left_wall_index: int
        right_wall_index: int
        surface_index: int

    nx: int
    nz: int

    def is_conforming(self, other: "Layer"):
        return self.nx == other.nx

    def generate_layer(
        self,
        boundary_below: "LayerBoundary.BuildResult",
        boundary_above: "LayerBoundary.BuildResult",
        gmsh: GmshContext,
    ) -> "Layer.BuildResult":
        # join boundary_below and boundary_above with left and right walls:
        # above should use the *_copy variants.
        wall_right = gmsh.model.geo.add_line(
            boundary_below.right_vertex, boundary_above.right_vertex_copy
        )
        wall_left = gmsh.model.geo.add_line(
            boundary_above.left_vertex_copy, boundary_below.left_vertex
        )
        # line loop: needed for surface. from definitions,
        loop = gmsh.model.geo.add_curve_loop(
            [boundary_below.curve, wall_right, -boundary_above.curve_copy, wall_left]
        )
        surf = gmsh.model.geo.add_plane_surface([loop])

        # set resolution explicitly
        gmsh.model.geo.mesh.set_transfinite_curve(wall_left, self.nz + 1)
        gmsh.model.geo.mesh.set_transfinite_curve(wall_right, self.nz + 1)
        gmsh.model.geo.mesh.set_transfinite_curve(boundary_below.curve, self.nx + 1)
        gmsh.model.geo.mesh.set_transfinite_curve(
            boundary_above.curve_copy, self.nx + 1
        )
        gmsh.model.geo.mesh.set_transfinite_surface(surf)
        gmsh.model.geo.mesh.setRecombine(2, surf)  # quads

        return Layer.BuildResult(
            left_wall_index=wall_left, right_wall_index=wall_right, surface_index=surf
        )
EPS = 1e-6


class LayerBoundary(ABC):
    """Represents an interface spanning across the entire length of the domain or the top/bottom
    boundaries.
    """

    @dataclass(frozen=True)
    class BuildResult:
        """Stores gmsh tags relevant to the interface. curve is left-to-right."""

        left_vertex: int
        right_vertex: int
        curve: int

        _curve_copy: int = field(init=False, default=-1)
        _left_vertex_copy: int = field(init=False, default=-1)
        _right_vertex_copy: int = field(init=False, default=-1)

        def initialize_curve_copy(
            self,
            layer_below: Layer | None,
            layer_above: Layer | None,
            gmsh: GmshContext,
        ):
            # duplicate curve only if we desire/need nonconformity:
            if (
                layer_above is None
                or layer_below is None
                or layer_above.is_conforming(layer_below)
            ):
                object.__setattr__(self, "_curve_copy", self.curve)
                object.__setattr__(self, "_left_vertex_copy", self.left_vertex)
                object.__setattr__(self, "_right_vertex_copy", self.right_vertex)
                return

            ((_, ccpy),) = gmsh.model.geo.copy([(1, self.curve)])
            object.__setattr__(self, "_curve_copy", ccpy)
            gmsh.model.geo.synchronize()

            # no clean way to get the copies made of left_vertex and right_vertex except by
            # querying ccpy:
            (_, ep1), (_, ep2) = gmsh.model.get_boundary([(1, ccpy)])
            ep1x = gmsh.model.get_value(0, ep1, [])[0]
            ep2x = gmsh.model.get_value(0, ep2, [])[0]
            if ep1x > ep2x:
                ep1x, ep2x = ep2x, ep1x
            object.__setattr__(self, "_left_vertex_copy", ep1)
            object.__setattr__(self, "_right_vertex_copy", ep2)

        @property
        def curve_copy(self) -> int:
            if self._curve_copy < 0:
                raise RuntimeError(
                    "curve copy not initialized. use initialize_curve_copy() first!"
                )
            return self._curve_copy

        @property
        def left_vertex_copy(self) -> int:
            if self._curve_copy < 0:
                raise RuntimeError(
                    "curve copy not initialized. use initialize_curve_copy() first!"
                )
            return self._left_vertex_copy

        @property
        def right_vertex_copy(self) -> int:
            if self._curve_copy < 0:
                raise RuntimeError(
                    "curve copy not initialized. use initialize_curve_copy() first!"
                )
            return self._right_vertex_copy

    @abstractmethod
    def build_layer(
        self, xlow: float, xhigh: float, gmsh: GmshContext
    ) -> "LayerBoundary.BuildResult": ...
