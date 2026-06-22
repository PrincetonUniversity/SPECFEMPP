from dataclasses import dataclass, field

import numpy as np

from ..binary_detect_N3 import faces_intersect, sample_face_jacobian
from .boundary import BoundarySpec
from .faces import FaceType, QUA_9_facecoords_at_node


@dataclass
class NonconformingInterfaces:
    """Stores nonconforming interfaces in a struct-of-arrays format."""

    elements_a: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.int32)
    )
    elements_b: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.int32)
    )
    faces_a: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.uint8))
    faces_b: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.uint8))

    def concatenate(self, other: "NonconformingInterfaces"):
        self.elements_a = np.concatenate([self.elements_a, other.elements_a])
        self.elements_b = np.concatenate([self.elements_b, other.elements_b])
        self.faces_a = np.concatenate([self.faces_a, other.faces_a])
        self.faces_b = np.concatenate([self.faces_b, other.faces_b])

    @staticmethod
    def join(
        a: "NonconformingInterfaces", b: "NonconformingInterfaces"
    ) -> "NonconformingInterfaces":
        return NonconformingInterfaces(
            elements_a=np.concatenate([a.elements_a, b.elements_a]),
            elements_b=np.concatenate([a.elements_b, b.elements_b]),
            faces_a=np.concatenate([a.faces_a, b.faces_a]),
            faces_b=np.concatenate([a.faces_b, b.faces_b]),
        )

    @staticmethod
    def from_boundaryspec(
        bdspec: BoundarySpec,
        node_locs: np.ndarray,
        element_nodes: np.ndarray,
    ) -> "NonconformingInterfaces":
        return _interfaces_from_boundaryspec(
            bdspec=bdspec,
            node_locs=node_locs,
            element_nodes=element_nodes,
        )


def _interfaces_from_boundaryspec(
    bdspec: BoundarySpec,
    node_locs: np.ndarray,
    element_nodes: np.ndarray,
) -> NonconformingInterfaces:
    """Finds intersecting edges between two boundary segments,
    returning a NonconformingInterfaces instance storing the results.

    Args:
        bdspec (BoundarySpec): Boundary data
        node_locs (np.ndarray): array of locations for each node
        element_nodes (np.ndarray): node indices for each element

    Returns:
        NonconformingInterfaces: The resulting interfaces, with `a`
            from entity1 and `b` from entity2.
    """

    # result arrays to build up
    nonconform_ispec = []
    nonconform_jspec = []
    nonconform_itype = []
    nonconform_jtype = []

    for item_a in bdspec.rtree.intersection(bdspec.rtree.bounds, objects=True):
        bd_a = item_a.id
        bbox_a = item_a.bbox
        elem_a: int = bdspec.element_inds[bd_a]  # type: ignore
        type_a: int = bdspec.element_faces[bd_a]  # type: ignore
        facenodes_a = element_nodes[
            elem_a, FaceType.HEX_27_node_indices_on_type(type_a)
        ]
        facenode_locs_a = node_locs[facenodes_a, :]
        for item_b in bdspec.rtree.intersection(bbox_a, objects=True):
            bd_b = item_b.id

            if bd_b < bd_a:  # we will only count intersections once.
                continue

            elem_b: int = bdspec.element_inds[bd_b]  # type: ignore
            type_b: int = bdspec.element_faces[bd_b]  # type: ignore
            facenodes_b = element_nodes[
                elem_b, FaceType.HEX_27_node_indices_on_type(type_b)
            ]
            facenode_locs_b = node_locs[facenodes_b, :]

            # edge case (lol): an element should not intersect itself on the same side
            # this might happen if the same boundary is used for two different surfaces.

            # we want to allow entities to be the same, since an entity can be a union
            # of others, so it may be that two elements on the same entity intersect

            if elem_a == elem_b and type_a == type_b:
                continue

            # likely scenario: these faces may share an edge or corner. If so, then skip the
            # point-localization
            matching_localcoords = None
            for imatch_face in np.where(np.isin(facenodes_a, facenodes_b))[0]:
                localcoord_a = QUA_9_facecoords_at_node(imatch_face)
                for jmatch_face in np.where(facenodes_b == facenodes_a[imatch_face])[0]:
                    localcoord_b = QUA_9_facecoords_at_node(jmatch_face)
                    matching_localcoords = (localcoord_a, localcoord_b)
                    break
                if matching_localcoords is not None:
                    break

            if faces_intersect(
                facenode_locs_a,
                facenode_locs_b,
                matching_localcoords=matching_localcoords,
            ):
                nonconform_ispec.append(elem_a)
                nonconform_jspec.append(elem_b)
                nonconform_itype.append(type_a)
                nonconform_jtype.append(type_b)

    return NonconformingInterfaces(
        elements_a=np.array(nonconform_ispec, dtype=np.int32),
        elements_b=np.array(nonconform_jspec, dtype=np.int32),
        faces_a=np.array(nonconform_itype, dtype=np.uint8),
        faces_b=np.array(nonconform_jtype, dtype=np.uint8),
    )
