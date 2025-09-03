from dataclasses import dataclass, field
from dataclasses import replace as dataclass_replace
from enum import IntEnum

import numpy as np


class EdgeType(IntEnum):
    BOTTOM = 0
    RIGHT = 1
    TOP = 2
    LEFT = 3

    @staticmethod
    def QUA_9_node_indices_on_type(edgetype: int) -> tuple[int, int, int]:
        if edgetype == EdgeType.BOTTOM:
            return (0, 4, 1)
        if edgetype == EdgeType.RIGHT:
            return (1, 5, 2)
        if edgetype == EdgeType.TOP:
            return (2, 6, 3)
        if edgetype == EdgeType.LEFT:
            return (3, 7, 0)
        msg = f"`edgetype` (={edgetype}) must be an EdgeType"
        raise ValueError(msg)

    @staticmethod
    def QUA_9_edge_to_inds_matrix() -> np.ndarray:
        return np.array([EdgeType.QUA_9_node_indices_on_type(i) for i in range(4)])


@dataclass
class ConformingInterfaces:
    """Stores nonconforming interfaces in a struct-of-arrays format."""

    elements_adj: np.ndarray = field(
        default_factory=lambda: np.full((0, 4), -1, dtype=np.int32)
    )

    @property
    def nelem(self):
        return self.elements_adj.shape[0]

    @staticmethod
    def join(
        a: "ConformingInterfaces",
        b: "ConformingInterfaces",
        a_joined_elem_inds: np.ndarray,
        b_joined_elem_inds: np.ndarray,
    ) -> "ConformingInterfaces":
        if a.nelem == 0:
            return dataclass_replace(b)
        if b.nelem == 0:
            return dataclass_replace(a)
        assert a.nelem == a_joined_elem_inds.shape[0], (
            "`a_joined_elem_inds` must represent the elements in `a`. "
            "They do not represent the same number of elements."
        )
        assert b.nelem == b_joined_elem_inds.shape[0], (
            "`b_joined_elem_inds` must represent the elements in `b`. "
            "They do not represent the same number of elements."
        )
        remapped_a_adj = np.where(a.elements_adj >= 0, a_joined_elem_inds[a.elements_adj], -1)
        remapped_b_adj = np.where(b.elements_adj >= 0, b_joined_elem_inds[b.elements_adj], -1)
        nelem = max(np.max(a_joined_elem_inds), np.max(b_joined_elem_inds))+1
        elements_new_adj = np.full((nelem, 4), -1, dtype=np.int32)
        elements_new_adj[a_joined_elem_inds, :] = remapped_a_adj

        # for all of b's indices:
        #   if a did not give an index: give b's index.
        #   otherwise: set to -2 if a's and b's disagree.
        elements_new_adj[b_joined_elem_inds, :] = np.where(
            elements_new_adj[b_joined_elem_inds, :] == -1,
            remapped_b_adj,
            np.where(
                elements_new_adj[b_joined_elem_inds, :] == remapped_b_adj,
                remapped_b_adj,
                -2,
            ),
        )
        if np.any(elements_new_adj == -2):
            rte = RuntimeError(
                "When joining two `Interface`s, the index mapping "
                "used has `a` and `b` disagree."
            )
            rte.add_note(
                "An element has taken from both `a` and `b`, each providing a "
                "disagreeing value. Both have provided an adjacent element on "
                "the same side, but that element is not the same between the two."
            )
        return ConformingInterfaces(elements_new_adj)

    @staticmethod
    def from_element_node_matrix(
        element_node_matrix: np.ndarray,
    ) -> "ConformingInterfaces":
        nelem = element_node_matrix.shape[0]
        edges = edges_of_all_elements(element_node_matrix, True).reshape((-1,3))
        # edges[ielem*4 + edgetype, :] == edgenodes

        #sort edges
        edges_sortinds = np.lexsort(edges.T,axis=0)
        edges_sorted = edges[edges_sortinds,:]

        edge_unique, ind, counts = np.unique(
            edges_sorted,
            axis=0,
            return_index=True,
            return_inverse=False,
            return_counts=True,
        )
        if np.any(counts > 2):
            rte = RuntimeError(
                "When forming a `ConformingInterface` from `element_node_matrix`, "
                "At least one edge exists at least 3 times. The max amount of occurences of a "
                "single edge should be two."
            )
            raise rte

        # populate elements_adj according to equivalent edges
        elements_adj = np.full((nelem, 4), -1, dtype=np.int32)
        paired = ind[counts == 2]
        # (paired, paired+1) are equal -- invert the sort
        elem_a, edge_a = np.unravel_index(edges_sortinds[paired],(nelem,4))
        elem_b, edge_b = np.unravel_index(edges_sortinds[paired+1],(nelem,4))
        elements_adj[elem_a,edge_a] = elem_b
        elements_adj[elem_b,edge_b] = elem_a

        return ConformingInterfaces(elements_adj=elements_adj)


def edges_of_all_elements(
    element_node_matrix: np.ndarray, consistent_order: bool = False
) -> np.ndarray:
    """Takes the element -> node matrix of a set of QUA_9 elements, and returns
    the nodes for each edge on each element.

    If `consistent_order` is set, the nodes for each edge are ordered as [a b c],
    where a < c. This is important for detecting conforming edges. Otherwise, the
    order is made in the counter-clockwise direction in terms of the local-coordinate
    orientation.

    Args:
        element_node_matrix (np.ndarray): The element -> node matrix (N x 9)
        consistent_order (bool, optional): Whether the order of nodes for each edge
        is based on orientation or node order. Defaults to False.

    Returns:
        np.ndarray: (N x 4 x 3) array of nodes per edge of each element.
    """
    edges = element_node_matrix[:, EdgeType.QUA_9_edge_to_inds_matrix()]
    if consistent_order:
        return np.where(
            edges[:, :, 0, None] > edges[:, :, -1, None], np.flip(edges, axis=2), edges
        )
    else:
        return edges
