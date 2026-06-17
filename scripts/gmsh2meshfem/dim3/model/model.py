from dataclasses import dataclass, field
from dataclasses import replace as dataclass_replace
from typing import Iterable

import numpy as np

from ...gmsh_dep import GmshContext

# from .boundary import BoundarySpec
# from .edges import ConformingInterfaces
from ...helper.index_mapping import IndexMapping, JoinedIndexMapping

# from .nonconforming_interfaces import (
#     NonconformingInterfaces,
# )
# from .physical_group import (
#     NullPhysicalGroup,
#     PhysicalGroup,
#     UnionPhysicalGroup,
#     physical_group_from_name,
# )
# from .plotter import plot_model


# TODO: consider using some sort of joint node and element index mapping container during
# construction, which would simplify the process. In particular, `element_nodes`
# needs to be shared around a lot.
@dataclass
class Model:
    @staticmethod
    def from_meshed_volume(
        volume: list[int] | int,
        gmsh: GmshContext,
        physical_group_captures: Iterable[str] | None = None,
    ) -> "Model":
        """Given an initialized mesh in gmsh, constructs a Model
        that stores the data of a volume or collection of
        volumes with the given tag(s). The resulting Model is
        fully functional, even with a deactivated GmshContext.

        Args:
            volume (list[int] | int): gmsh volume tag(s)
            gmsh (GmshContext): the gmsh handshake to secure active environment.
            physical_group_captures (list[str] | None): a list of the physical groups to store.
        """
        if isinstance(volume, list):
            if len(volume) == 0:
                msg = "No volume tags specified. Cannot create a model."
                raise ValueError(msg)
            if len(volume) == 1:
                volume = volume[0]
        if isinstance(volume, int):
            # single volume, can be done manually
            meshnodes = gmsh.model.mesh.get_nodes()
            node_indexing = IndexMapping(meshnodes[0])
            node_locs = np.reshape(meshnodes[1], (-1, 3))

            # gmsh.model.mesh.get_elements gives elements of different types.
            # each of these captures a case:

            def on_mesh_tri(triname):
                msg = f"Cannot mesh {triname}. Must be quad."
                raise ValueError(msg)

            def on_MSH_QUA_4(elems, nodes):
                msg = (
                    "At the moment, 4-node quads have not been implemented. "
                    "Please mesh at order-2 by setting "
                    '`gmsh.option.setNumber("Mesh.ElementOrder", 2)`.'
                )
                raise NotImplementedError(msg)

            def on_MSH_QUA_8(elems, nodes):
                msg = (
                    "At the moment, 8-node quads have not been implemented. "
                    "The 9th node (center) must be placed manually."
                )
                raise NotImplementedError(msg)

            element_nodes_list = []
            layer_indices_list = []

            def on_MSH_QUA_9(elems, nodes):
                element_nodes_list.append(
                    node_indexing.apply(np.reshape(nodes, (-1, 9)))
                )
                layer_indices_list.append(
                    np.full(element_nodes_list[-1].shape[0], volume, dtype=np.uint8)
                )

            # https://gitlab.onelab.info/gmsh/gmsh/blob/master/src/common/GmshDefines.h
            gmsh.for_element_types_in_entity(
                2,
                volume,
                {
                    3: on_MSH_QUA_4,
                    16: on_MSH_QUA_8,
                    10: on_MSH_QUA_9,
                    2: lambda a, b: on_mesh_tri("3-node triangle"),
                    9: lambda a, b: on_mesh_tri("6-node 2nd order triangle"),
                    20: lambda a, b: on_mesh_tri(
                        "9-node 3rd order incomplete triangle"
                    ),
                    21: lambda a, b: on_mesh_tri("10-node 3rd order triangle"),
                    22: lambda a, b: on_mesh_tri(
                        "12-node 4th order incomplete triangle"
                    ),
                    23: lambda a, b: on_mesh_tri("15-node 4th order triangle"),
                    24: lambda a, b: on_mesh_tri(
                        "15-node 5th order incomplete triangle"
                    ),
                    25: lambda a, b: on_mesh_tri("21-node 5th order triangle"),
                },
            )
            element_nodes = np.concatenate(element_nodes_list, axis=0)
            layer_indices = np.concatenate(layer_indices_list, axis=0)

            boundary_entities = gmsh.model.get_boundary(
                [(2, volume)], oriented=False, recursive=False
            )

            physical_groups: dict[str, PhysicalGroup] = {}
            for name in (
                [] if physical_group_captures is None else physical_group_captures
            ):
                physical_groups[name] = physical_group_from_name(
                    gmsh,
                    node_indexing.invert(element_nodes),
                    node_indexing,
                    node_locs,
                    name,
                )

            return Model(
                nodes=node_locs,
                elements=element_nodes,
                materials=layer_indices,
                boundaries=BoundarySpec.from_model_entity(
                    gmsh,
                    [tag for dim, tag in boundary_entities if dim == 1],
                    node_indexing.invert(element_nodes),
                    node_indexing,
                    node_locs,
                ),
                _node_gmshtag_to_index_mapping=node_indexing,
                conforming_interfaces=ConformingInterfaces.from_element_node_matrix(
                    element_nodes
                ),
                physical_groups=physical_groups,
            )
        else:
            return Model.union(
                Model.from_meshed_volume(
                    volume[0], gmsh, physical_group_captures=physical_group_captures
                ),
                Model.from_meshed_volume(
                    volume[1:], gmsh, physical_group_captures=physical_group_captures
                ),
            )
