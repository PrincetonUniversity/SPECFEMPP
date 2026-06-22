import itertools

from gmsh2meshfem.dim2.model import Model
from gmsh2meshfem.gmsh_dep import GmshContext

from ..tags import BOUNDARY_TYPES, BoundaryConditionType
from .layer2d import Layer2D, LayerBoundary2D


class LayeredBuilder2D:
    """Generates a layer topography domain in 2D, spanning from x=xlow to x=xhigh.
    Each layer `layers[i]` is bounded below by `boundaries[i]` and above by `boundaries[i+1]`.
    """

    xlow: float
    xhigh: float

    boundaries: list[LayerBoundary2D]
    layers: list[Layer2D]

    domain_boundary_type_top: BoundaryConditionType
    domain_boundary_type_bottom: BoundaryConditionType
    domain_boundary_type_left: BoundaryConditionType
    domain_boundary_type_right: BoundaryConditionType

    @property
    def width(self):
        return self.xhigh - self.xlow

    def __init__(
        self,
        xlow: float,
        xhigh: float,
        set_left_boundary: BoundaryConditionType = "neumann",
        set_right_boundary: BoundaryConditionType = "neumann",
        set_top_boundary: BoundaryConditionType = "neumann",
        set_bottom_boundary: BoundaryConditionType = "neumann",
    ):
        self.xlow = xlow
        self.xhigh = xhigh
        self.layers = []
        self.boundaries = []
        self.domain_boundary_type_top = set_top_boundary
        self.domain_boundary_type_bottom = set_bottom_boundary
        self.domain_boundary_type_left = set_left_boundary
        self.domain_boundary_type_right = set_right_boundary

    def create_model(self) -> Model:
        with GmshContext() as gmsh:
            built_layerbds = [
                bdlayer.build_layer(self.xlow, self.xhigh, gmsh=gmsh)
                for bdlayer in self.boundaries
            ]
            for ilayer, layerbd in enumerate(built_layerbds):
                layerbd.initialize_curve_copy(
                    None if ilayer == 0 else self.layers[ilayer - 1],
                    None if ilayer == len(self.layers) else self.layers[-1],
                    gmsh,
                )

            # store tags
            surfaces = []
            left_walls = []
            right_walls = []
            for i, (l0, l1) in enumerate(itertools.pairwise(built_layerbds)):
                layer_result = self.layers[i].generate_layer(l0, l1, gmsh)
                surfaces.append(layer_result.surface_index)
                left_walls.append(layer_result.left_wall_index)
                right_walls.append(layer_result.right_wall_index)

            # physical groups in model space, but our geometry construction is
            # currently only in geo space. Sync so physical groups can access
            # entities
            gmsh.model.geo.synchronize()

            # set physical groups for 4 sides. These aren't used by Model,
            # but may be useful for future implementation.
            # We will select from these physical groups when setting BCs
            bottom_floor = built_layerbds[0].curve
            top_ceiling = built_layerbds[-1].curve_copy

            gmsh.model.add_physical_group(1, left_walls, name="left_boundary")
            gmsh.model.add_physical_group(1, right_walls, name="right_boundary")
            gmsh.model.add_physical_group(1, [bottom_floor], name="bottom_boundary")
            gmsh.model.add_physical_group(1, [top_ceiling], name="top_boundary")

            # append edge tags to these arrays
            # we will physical group afterwards
            bdry_by_name = {condition: [] for condition in BOUNDARY_TYPES}

            for layer, leftwall, rightwall in zip(
                self.layers, left_walls, right_walls, strict=True
            ):
                # add left and right to boundaries, with exception of skipping AFS when desired
                if not (
                    self.domain_boundary_type_left == "acoustic_free_surface"
                    and layer.skip_acoustic_free_surface
                ):
                    bdry_by_name[self.domain_boundary_type_left].append(leftwall)
                if not (
                    self.domain_boundary_type_right == "acoustic_free_surface"
                    and layer.skip_acoustic_free_surface
                ):
                    bdry_by_name[self.domain_boundary_type_right].append(rightwall)

            # same for top and bottom: add to bdries, except a skipped AFS
            if not (
                self.domain_boundary_type_bottom == "acoustic_free_surface"
                and self.layers[0].skip_acoustic_free_surface
            ):
                bdry_by_name[self.domain_boundary_type_bottom].append(bottom_floor)
            if not (
                self.domain_boundary_type_top == "acoustic_free_surface"
                and self.layers[-1].skip_acoustic_free_surface
            ):
                bdry_by_name[self.domain_boundary_type_top].append(top_ceiling)

            # boundary condition marking complete: give info to gmsh

            # set physical group
            for name, bdry in bdry_by_name.items():
                if bdry:
                    gmsh.model.add_physical_group(1, bdry, name=name)

            # required for ngnod = 9
            gmsh.option.setNumber("Mesh.ElementOrder", 2)
            gmsh.model.mesh.generate()

            # === uncomment this to see GUI ===
            # gmsh.fltk.run()

            # =====================================================================
            #                      extract mesh model
            # =====================================================================
            return Model.from_meshed_surface(
                surface=surfaces,
                gmsh=gmsh,
                physical_group_captures=bdry_by_name.keys(),
            )
