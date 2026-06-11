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
        left_wall_index: int
        right_wall_index: int
        front_wall_index: int
        back_wall_index: int
        volume_index: int

    nx: int
    ny: int
    nz: int
    skip_acoustic_free_surface: bool = False

    def is_conforming(self, other: "Layer3D"):
        return self.nx == other.nx and self.ny == other.ny

    def generate_layer(
        self,
        boundary_below: "LayerBoundary3D.BuildResult",
        boundary_above: "LayerBoundary3D.BuildResult",
        gmsh: GmshContext,
    ) -> "Layer3D.BuildResult":
        # join boundary_below and boundary_above with left and right walls:
        # above should use the *_copy variants.

        line_front_left = gmsh.model.geo.add_line(
            boundary_below.corner_front_left, boundary_above.corner_front_left_copy
        )
        line_front_right = gmsh.model.geo.add_line(
            boundary_below.corner_front_right, boundary_above.corner_front_right_copy
        )
        line_back_left = gmsh.model.geo.add_line(
            boundary_below.corner_back_left, boundary_above.corner_back_left_copy
        )
        line_back_right = gmsh.model.geo.add_line(
            boundary_below.corner_back_right, boundary_above.corner_back_right_copy
        )

        curveloop = gmsh.model.geo.add_curve_loop(
            [
                boundary_below.curve_front,
                line_front_right,
                -boundary_above.curve_front_copy,
                -line_front_left,
            ]
        )
        front_wall = gmsh.model.geo.add_plane_surface([curveloop])

        curveloop = gmsh.model.geo.add_curve_loop(
            [
                -boundary_below.curve_back,
                line_back_left,
                boundary_above.curve_back_copy,
                -line_back_right,
            ]
        )
        back_wall = gmsh.model.geo.add_plane_surface([curveloop])

        curveloop = gmsh.model.geo.add_curve_loop(
            [
                boundary_below.curve_right,
                line_back_right,
                -boundary_above.curve_right_copy,
                -line_front_right,
            ]
        )
        right_wall = gmsh.model.geo.add_plane_surface([curveloop])

        curveloop = gmsh.model.geo.add_curve_loop(
            [
                -boundary_below.curve_back,
                line_front_left,
                boundary_above.curve_back_copy,
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
                boundary_above.surface_copy,
                boundary_below.surface,
            ]
        )
        volume = gmsh.model.geo.add_volume([surfloop])

        # set resolution explicitly
        for curve in [
            boundary_below.curve_front,
            boundary_above.curve_front_copy,
            boundary_below.curve_back,
            boundary_above.curve_back_copy,
        ]:
            gmsh.model.mesh.set_transfinite_curve(curve, self.nx + 1)
        for curve in [
            boundary_below.curve_left,
            boundary_above.curve_left_copy,
            boundary_below.curve_right,
            boundary_above.curve_right_copy,
        ]:
            gmsh.model.mesh.set_transfinite_curve(curve, self.ny + 1)
        for curve in [
            line_front_left,
            line_front_right,
            line_back_right,
            line_back_left,
        ]:
            gmsh.model.mesh.set_transfinite_curve(curve, self.nz + 1)
        for surf in [front_wall, right_wall, back_wall, left_wall]:
            gmsh.model.geo.mesh.set_transfinite_surface(surf)
            gmsh.model.geo.mesh.setRecombine(2, surf)  # quads
            gmsh.model.mesh.set_smoothing(2, surf, 30)  # relax quads to be more regular

        gmsh.model.mesh.set_transfinite_volume(volume)

        return Layer3D.BuildResult(
            left_wall_index=left_wall,
            right_wall_index=right_wall,
            front_wall_index=front_wall,
            back_wall_index=back_wall,
            volume_index=volume,
        )


class LayerBoundary3D(ABC):
    """Represents an interface spanning across the entire length of the domain or the top/bottom
    boundaries.
    """

    @dataclass(frozen=True)
    class BuildResult:
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

        _copy_corner_front_left: int = field(init=False, default=0)
        _copy_corner_front_right: int = field(init=False, default=0)
        _copy_corner_back_right: int = field(init=False, default=0)
        _copy_corner_back_left: int = field(init=False, default=0)

        _copy_curve_front: int = field(init=False, default=0)
        _copy_curve_right: int = field(init=False, default=0)
        _copy_curve_back: int = field(init=False, default=0)
        _copy_curve_left: int = field(init=False, default=0)

        _copy_surface: int = field(init=False, default=0)

        def initialize_copy(
            self,
            layer_below: Layer3D | None,
            layer_above: Layer3D | None,
            gmsh: GmshContext,
        ):
            # duplicate curve only if we desire/need nonconformity:
            if (
                layer_above is None or layer_below is None
                # for now, don't repeat geometry -- we will just assume users want nonconforming.
                # uncomment this line below:
                # or layer_above.is_conforming(layer_below)
            ):
                # prevent duplication by setting copied entity fields to reference original.
                for fieldname in [
                    "surface",
                    "curve_front",
                    "curve_right",
                    "curve_back",
                    "curve_left",
                    "corner_front_left",
                    "corner_front_right",
                    "corner_back_right",
                    "corner_back_left",
                ]:
                    object.__setattr__(
                        self, f"_copy_{fieldname}", getattr(self, fieldname)
                    )

                return
            ((_, surface_copy),) = gmsh.model.geo.copy([(2, self.surface)])
            object.__setattr__(self, "_copy_surface", surface_copy)

            gmsh.model.geo.synchronize()

            # no clean way to get the copies of boundaries except by
            # querying get_boundary:
            bd_dimtags = gmsh.model.get_boundary([(2, surface_copy)], recursive=True)
            curve_sample_coord = 1

            # tag and coordinates
            curves = {
                tag: gmsh.model.get_value(1, tag, [curve_sample_coord])
                for dim, tag in bd_dimtags
                if dim == 1
            }
            verts = {
                tag: gmsh.model.get_value(0, tag, [])
                for dim, tag in bd_dimtags
                if dim == 0
            }

            for fieldname, is_curve in [
                ("curve_front", True),
                ("curve_right", True),
                ("curve_back", True),
                ("curve_left", True),
                ("corner_front_left", False),
                ("corner_front_right", False),
                ("corner_back_right", False),
                ("corner_back_left", False),
            ]:
                # get closest
                matchtag = 0
                dist_val = np.inf

                orig_tag = getattr(self, fieldname)
                if is_curve:
                    thisloc = gmsh.model.get_value(1, orig_tag, [curve_sample_coord])
                else:
                    thisloc = gmsh.model.get_value(0, orig_tag, [])

                for sampletag, pt in curves if is_curve else verts:
                    if np.linalg.norm(pt - thisloc) < dist_val:
                        matchtag = sampletag

                assert dist_val < 1e-5, (
                    f"When matching boundary entity {fieldname} ({orig_tag}) of cloned surface:"
                    f" matched entity ({matchtag}) is distance {dist_val}, which should be 0."
                )
                object.__setattr__(self, f"_copy_{fieldname}", matchtag)

                if is_curve:
                    del curves[matchtag]
                else:
                    del verts[matchtag]

        @property
        def curve_front_copy(self) -> int:
            if self._copy_curve_front < 0:
                raise RuntimeError(
                    "curve copy not initialized. use initialize_copy() first!"
                )
            return self._copy_curve_front

        @property
        def curve_back_copy(self) -> int:
            if self._copy_curve_back < 0:
                raise RuntimeError(
                    "curve copy not initialized. use initialize_copy() first!"
                )
            return self._copy_curve_back

        @property
        def curve_left_copy(self) -> int:
            if self._copy_curve_left < 0:
                raise RuntimeError(
                    "curve copy not initialized. use initialize_copy() first!"
                )
            return self._copy_curve_left

        @property
        def curve_right_copy(self) -> int:
            if self._copy_curve_right < 0:
                raise RuntimeError(
                    "curve copy not initialized. use initialize_copy() first!"
                )
            return self._copy_curve_right

        @property
        def corner_back_left_copy(self) -> int:
            if self._copy_corner_back_left < 0:
                raise RuntimeError(
                    "curve copy not initialized. use initialize_copy() first!"
                )
            return self._copy_corner_back_left

        @property
        def corner_back_right_copy(self) -> int:
            if self._copy_corner_back_right < 0:
                raise RuntimeError(
                    "curve copy not initialized. use initialize_copy() first!"
                )
            return self._copy_corner_back_right

        @property
        def corner_front_left_copy(self) -> int:
            if self._copy_corner_front_left < 0:
                raise RuntimeError(
                    "curve copy not initialized. use initialize_copy() first!"
                )
            return self._copy_corner_front_left

        @property
        def corner_front_right_copy(self) -> int:
            if self._copy_corner_front_right < 0:
                raise RuntimeError(
                    "curve copy not initialized. use initialize_copy() first!"
                )
            return self._copy_corner_front_right

        @property
        def surface_copy(self) -> int:
            if self._copy_surface < 0:
                raise RuntimeError(
                    "curve copy not initialized. use initialize_copy() first!"
                )
            return self._copy_surface

    @abstractmethod
    def build_layer(
        self, xlow: float, xhigh: float, ylow: float, yhigh: float, gmsh: GmshContext
    ) -> "LayerBoundary3D.BuildResult": ...
