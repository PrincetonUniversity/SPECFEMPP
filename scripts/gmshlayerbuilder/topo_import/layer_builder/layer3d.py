from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from gmsh2meshfem.gmsh_dep import GmshContext


@dataclass
class Layer3D:
    """A "layer" denoes a region spanning the width of the domain between two `LayerBoundary`s.
    Each Layer is meshed as a deformed grid (gmsh-transfinite) with specified number of cells
    along the horizontal axes (nx, ny) and vertical axis (nz).
    """

    @dataclass
    class BuildResult:
        nx: int
        ny: int
        nz: int

        curve_top_front_index: int
        curve_bottom_front_index: int
        curve_top_back_index: int
        curve_bottom_back_index: int
        curve_top_left_index: int
        curve_bottom_left_index: int
        curve_top_right_index: int
        curve_bottom_right_index: int

        line_front_left_index: int
        line_front_right_index: int
        line_back_right_index: int
        line_back_left_index: int

        left_wall_index: int
        right_wall_index: int
        front_wall_index: int
        back_wall_index: int
        top_surface_index: int
        bottom_surface_index: int

        volume_index: int

        def update_mesh_params(self, gmsh: GmshContext):
            # set resolution explicitly
            for curve in [
                self.curve_top_front_index,
                self.curve_bottom_front_index,
                self.curve_top_back_index,
                self.curve_bottom_back_index,
            ]:
                gmsh.model.mesh.set_transfinite_curve(curve, self.nx + 1)
            for curve in [
                self.curve_top_left_index,
                self.curve_bottom_left_index,
                self.curve_top_right_index,
                self.curve_bottom_right_index,
            ]:
                gmsh.model.mesh.set_transfinite_curve(curve, self.ny + 1)
            for curve in [
                self.line_front_left_index,
                self.line_front_right_index,
                self.line_back_right_index,
                self.line_back_left_index,
            ]:
                gmsh.model.mesh.set_transfinite_curve(curve, self.nz + 1)
            for surf in [
                self.front_wall_index,
                self.right_wall_index,
                self.back_wall_index,
                self.left_wall_index,
                self.top_surface_index,
                self.bottom_surface_index,
            ]:
                gmsh.model.mesh.set_transfinite_surface(surf)
                gmsh.model.mesh.setRecombine(2, surf)  # quads
                gmsh.model.mesh.set_smoothing(
                    2, surf, 30
                )  # relax quads to be more regular

            gmsh.model.mesh.set_transfinite_volume(self.volume_index)

    nx: int
    ny: int
    nz: int
    skip_acoustic_free_surface: bool = False

    def is_conforming(self, other: "Layer3D"):
        return self.nx == other.nx and self.ny == other.ny

    def generate_layer_geometry(
        self,
        boundary_below: "LayerBoundary3D.BuildResult",
        boundary_above: "LayerBoundary3D.BuildResult",
        gmsh: GmshContext,
    ) -> "Layer3D.BuildResult":
        # join boundary_below and boundary_above with left and right walls:
        # above should use the *_copy variants.

        line_front_left = gmsh.model.geo.add_line(
            boundary_below.above.corner_front_left,
            boundary_above.below.corner_front_left,
        )
        line_front_right = gmsh.model.geo.add_line(
            boundary_below.above.corner_front_right,
            boundary_above.below.corner_front_right,
        )
        line_back_left = gmsh.model.geo.add_line(
            boundary_below.above.corner_back_left, boundary_above.below.corner_back_left
        )
        line_back_right = gmsh.model.geo.add_line(
            boundary_below.above.corner_back_right,
            boundary_above.below.corner_back_right,
        )

        curveloop = gmsh.model.geo.add_curve_loop(
            [
                boundary_below.above.curve_front,
                line_front_right,
                -boundary_above.below.curve_front,
                -line_front_left,
            ]
        )
        front_wall = gmsh.model.geo.add_plane_surface([curveloop])

        curveloop = gmsh.model.geo.add_curve_loop(
            [
                -boundary_below.above.curve_back,
                line_back_left,
                boundary_above.below.curve_back,
                -line_back_right,
            ]
        )
        back_wall = gmsh.model.geo.add_plane_surface([curveloop])

        curveloop = gmsh.model.geo.add_curve_loop(
            [
                boundary_below.above.curve_right,
                line_back_right,
                -boundary_above.below.curve_right,
                -line_front_right,
            ]
        )
        right_wall = gmsh.model.geo.add_plane_surface([curveloop])

        curveloop = gmsh.model.geo.add_curve_loop(
            [
                -boundary_below.above.curve_left,
                line_front_left,
                boundary_above.below.curve_left,
                -line_back_left,
            ]
        )
        left_wall = gmsh.model.geo.add_plane_surface([curveloop])

        surfloop = gmsh.model.geo.add_surface_loop(
            [
                front_wall,
                right_wall,
                back_wall,
                left_wall,
                boundary_above.below.surface,
                boundary_below.above.surface,
            ]
        )
        volume = gmsh.model.geo.add_volume([surfloop])

        return Layer3D.BuildResult(
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            curve_top_front_index=boundary_above.below.curve_front,
            curve_bottom_front_index=boundary_below.above.curve_front,
            curve_top_back_index=boundary_above.below.curve_back,
            curve_bottom_back_index=boundary_below.above.curve_back,
            curve_top_left_index=boundary_above.below.curve_left,
            curve_bottom_left_index=boundary_below.above.curve_left,
            curve_top_right_index=boundary_above.below.curve_right,
            curve_bottom_right_index=boundary_below.above.curve_right,
            line_front_left_index=line_front_left,
            line_back_left_index=line_back_left,
            line_front_right_index=line_front_right,
            line_back_right_index=line_back_right,
            left_wall_index=left_wall,
            right_wall_index=right_wall,
            front_wall_index=front_wall,
            top_surface_index=boundary_above.below.surface,
            bottom_surface_index=boundary_below.above.surface,
            back_wall_index=back_wall,
            volume_index=volume,
        )


class LayerBoundary3D(ABC):
    """Represents an interface spanning across the entire length of the domain or the top/bottom
    boundaries.
    """

    @dataclass(frozen=True)
    class BuildResultPart:
        """Stores gmsh tags relevant to the interface. curve is directed from lower coordinate
        value to upper."""

        corner_front_left: int
        corner_front_right: int
        corner_back_right: int
        corner_back_left: int

        curve_front: int
        curve_right: int
        curve_back: int
        curve_left: int

        surface: int

    @dataclass(frozen=True)
    class BuildResult:
        """Stores gmsh tags relevant to the interface. curve is directed from lower coordinate
        value to upper."""

        below: "LayerBoundary3D.BuildResultPart" = None  # type: ignore
        above: "LayerBoundary3D.BuildResultPart" = None  # type: ignore

        def __post_init__(self):
            # in case only one was passed in, set the other to same reference
            if self.below is None and self.above is None:
                e = RuntimeError("BuildResult requires at least one side set.")
                raise e

            if self.below is None:
                object.__setattr__(self, "below", self.above)
            if self.above is None:
                object.__setattr__(self, "above", self.below)

    @abstractmethod
    def build_layer(
        self,
        xlow: float,
        xhigh: float,
        ylow: float,
        yhigh: float,
        nonconforming_above_and_below: bool,
        gmsh: GmshContext,
    ) -> "LayerBoundary3D.BuildResult": ...
