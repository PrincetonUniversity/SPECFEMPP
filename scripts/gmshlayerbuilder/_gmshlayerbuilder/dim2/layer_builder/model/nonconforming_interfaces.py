from dataclasses import dataclass, field

from _gmshlayerbuilder.gmsh_dep import GmshContext
import numpy as np

from _gmshlayerbuilder.dim2.binary_detect_N3 import quadratic_beziers_intersect
from .boundary import BoundarySpec
from .index_mapping import IndexMapping
from .edges import edges_of_all_elements, EdgeType


@dataclass
class NonconformingInterfaces:
    """Stores nonconforming interfaces in a struct-of-arrays format."""

    elements_a: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.int32)
    )
    elements_b: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.int32)
    )
    edges_a: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.uint8))
    edges_b: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.uint8))

    def concatenate(self, other: "NonconformingInterfaces"):
        self.elements_a = np.concatenate([self.elements_a, other.elements_a])
        self.elements_b = np.concatenate([self.elements_b, other.elements_b])
        self.edges_a = np.concatenate([self.edges_a, other.edges_a])
        self.edges_b = np.concatenate([self.edges_b, other.edges_b])

    @staticmethod
    def join(
        a: "NonconformingInterfaces", b: "NonconformingInterfaces"
    ) -> "NonconformingInterfaces":
        return NonconformingInterfaces(
            elements_a=np.concatenate([a.elements_a, b.elements_a]),
            elements_b=np.concatenate([a.elements_b, b.elements_b]),
            edges_a=np.concatenate([a.edges_a, b.edges_a]),
            edges_b=np.concatenate([a.edges_b, b.edges_b]),
        )

    @staticmethod
    def between_entities(
        bdspec: BoundarySpec,
        entity1: int | BoundarySpec.EntityKey,
        entity2: int | BoundarySpec.EntityKey,
        node_locs: np.ndarray,
        element_nodes: np.ndarray,
    ) -> "NonconformingInterfaces":
        return interfaces_from_boundaryspec_entities(
            bdspec=bdspec,
            entity1=entity1,
            entity2=entity2,
            node_locs=node_locs,
            element_nodes=element_nodes,
        )


def interfaces_from_boundaryspec_entities(
    bdspec: BoundarySpec,
    entity1: int | BoundarySpec.EntityKey,
    entity2: int | BoundarySpec.EntityKey,
    node_locs: np.ndarray,
    element_nodes: np.ndarray,
) -> NonconformingInterfaces:
    nonconform_ispec = []
    nonconform_jspec = []
    nonconform_iedge = []
    nonconform_jedge = []

    edges_to_nodes = edges_of_all_elements(element_nodes)

    if isinstance(entity1, int):
        entity1 = bdspec.boundary_entity_spec[entity1]
    if isinstance(entity2, int):
        entity2 = bdspec.boundary_entity_spec[entity2]

    for i_a in range(entity1.start_index, entity1.end_index):
        elem_a = bdspec.element_inds[i_a]
        edge_a = bdspec.element_edges[i_a]

        edgenodes_a = edges_to_nodes[elem_a, edge_a, :]
        for i_b in range(entity2.start_index, entity2.end_index):
            elem_b = bdspec.element_inds[i_b]
            edge_b = bdspec.element_edges[i_b]
            edgenodes_b = edges_to_nodes[elem_b, edge_b, :]

            if quadratic_beziers_intersect(
                node_locs[edgenodes_a, ::2], node_locs[edgenodes_b, ::2]
            ):
                nonconform_ispec.append(elem_a)
                nonconform_jspec.append(elem_b)
                nonconform_iedge.append(edge_a)
                nonconform_jedge.append(edge_b)

    return NonconformingInterfaces(
        elements_a=np.array(nonconform_ispec, dtype=np.int32),
        elements_b=np.array(nonconform_jspec, dtype=np.int32),
        edges_a=np.array(nonconform_iedge, dtype=np.uint8),
        edges_b=np.array(nonconform_jedge, dtype=np.uint8),
    )
