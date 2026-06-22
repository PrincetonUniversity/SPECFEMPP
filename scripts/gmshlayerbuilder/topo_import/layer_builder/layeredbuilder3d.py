import itertools

from gmsh2meshfem.dim3.model.model import Model
from gmsh2meshfem.dim3.model.gmshmodel import GmshModel3D
from gmsh2meshfem.gmsh_dep import GmshContext

from ..tags import BOUNDARY_TYPES, BoundaryConditionType
from .layer3d import Layer3D, LayerBoundary3D


class LayeredBuilder3D:
    """Generates a layer topography domain in 3D, spanning from x=xlow, y=ylow to x=xhigh y=yhigh.
    Each layer `layers[i]` is bounded below by `boundaries[i]` and above by `boundaries[i+1]`.
    """

    xlow: float
    xhigh: float
    ylow: float
    yhigh: float

    boundaries: list[LayerBoundary3D]
    layers: list[Layer3D]

    domain_boundary_type_top: BoundaryConditionType
    domain_boundary_type_bottom: BoundaryConditionType
    domain_boundary_type_left: BoundaryConditionType
    domain_boundary_type_right: BoundaryConditionType
    domain_boundary_type_front: BoundaryConditionType
    domain_boundary_type_back: BoundaryConditionType

    @property
    def width_x(self):
        return self.xhigh - self.xlow

    @property
    def width_y(self):
        return self.yhigh - self.ylow

    def __init__(
        self,
        xlow: float,
        xhigh: float,
        ylow: float,
        yhigh: float,
        set_left_boundary: BoundaryConditionType = "neumann",
        set_right_boundary: BoundaryConditionType = "neumann",
        set_top_boundary: BoundaryConditionType = "neumann",
        set_bottom_boundary: BoundaryConditionType = "neumann",
        set_front_boundary: BoundaryConditionType = "neumann",
        set_back_boundary: BoundaryConditionType = "neumann",
    ):
        self.xlow = xlow
        self.xhigh = xhigh
        self.ylow = ylow
        self.yhigh = yhigh
        self.layers = []
        self.boundaries = []
        self.domain_boundary_type_top = set_top_boundary
        self.domain_boundary_type_bottom = set_bottom_boundary
        self.domain_boundary_type_left = set_left_boundary
        self.domain_boundary_type_right = set_right_boundary
        self.domain_boundary_type_front = set_front_boundary
        self.domain_boundary_type_back = set_back_boundary

    def create_model(self) -> Model:
        with GmshContext() as gmsh:
            # generate geometry of layer boundaries in gmsh
            built_layerbds = [
                bdlayer.build_layer(
                    self.xlow,
                    self.xhigh,
                    self.ylow,
                    self.yhigh,
                    ilayer > 0
                    and ilayer
                    < len(self.boundaries)
                    - 1,  # assume nonconformity for all (except top and bottom boundaries)
                    gmsh=gmsh,
                )
                for ilayer, bdlayer in enumerate(self.boundaries)
            ]

            # clean up node formation (assign boundary nodes to their respective entities)
            gmsh.model.mesh.reclassifyNodes()
            # generate geometric entities from discrete (mesh) entities
            gmsh.model.mesh.createGeometry()

            # store tags
            volumes = []
            left_walls = []
            right_walls = []
            front_walls = []
            back_walls = []

            layer_results = []
            for i, (l0, l1) in enumerate(itertools.pairwise(built_layerbds)):
                layer_result = self.layers[i].generate_layer_geometry(l0, l1, gmsh)
                layer_results.append(layer_result)
                volumes.append(layer_result.volume_index)
                left_walls.append(layer_result.left_wall_index)
                right_walls.append(layer_result.right_wall_index)
                front_walls.append(layer_result.front_wall_index)
                back_walls.append(layer_result.back_wall_index)

            gmsh.model.geo.synchronize()

            for layer_result in layer_results:
                layer_result.update_mesh_params(gmsh)

            # set physical groups for 4 sides. These aren't used by Model,
            # but may be useful for future implementation.
            # We will select from these physical groups when setting BCs
            bottom_floor = built_layerbds[0].above.surface
            top_ceiling = built_layerbds[-1].below.surface

            gmsh.model.add_physical_group(1, left_walls, name="left_boundary")
            gmsh.model.add_physical_group(1, right_walls, name="right_boundary")
            gmsh.model.add_physical_group(1, [bottom_floor], name="bottom_boundary")
            gmsh.model.add_physical_group(1, [top_ceiling], name="top_boundary")

            # append edge tags to these arrays
            # we will physical group afterwards
            bdry_by_name = {condition: [] for condition in BOUNDARY_TYPES}

            for layer, leftwall, rightwall, frontwall, backwall in zip(
                self.layers,
                left_walls,
                right_walls,
                front_walls,
                back_walls,
                strict=True,
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
                if not (
                    self.domain_boundary_type_front == "acoustic_free_surface"
                    and layer.skip_acoustic_free_surface
                ):
                    bdry_by_name[self.domain_boundary_type_front].append(frontwall)
                if not (
                    self.domain_boundary_type_back == "acoustic_free_surface"
                    and layer.skip_acoustic_free_surface
                ):
                    bdry_by_name[self.domain_boundary_type_back].append(backwall)

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

            # required for ngnod = 27
            gmsh.option.setNumber("Mesh.ElementOrder", 2)
            gmsh.model.mesh.generate()

            # === uncomment this to see GUI ===
            # gmsh.fltk.run()

            # =====================================================================
            #                      extract mesh model
            # =====================================================================
            return GmshModel3D(gmsh,volumes).to_model({v:v for v in volumes})
            # return Model.from_meshed_volume(
            #     volume=volumes,
            #     gmsh=gmsh,
            #     physical_group_captures=bdry_by_name.keys(),
            # )
