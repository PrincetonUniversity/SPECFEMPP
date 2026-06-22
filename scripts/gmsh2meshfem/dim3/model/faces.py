from dataclasses import dataclass, field
from dataclasses import replace as dataclass_replace
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from ...dim2.binary_detect_N3 import L, maxfind_coefs_a, maxfind_coefs_b
from ...dim2.model.edges import EdgeType

# https://gmsh.info/doc/texinfo/
# Hexahedron:             Hexahedron20:          Hexahedron27:
#
#        v
# 3----------2            3----13----2           3----13----2
# |\     ^   |\           |\         |\          |\         |\
# | \    |   | \          | 15       | 14        |15    24  | 14
# |  \   |   |  \         9  \       11 \        9  \ 20    11 \
# |   7------+---6        |   7----19+---6       |   7----19+---6
# |   |  +-- |-- | -> u   |   |      |   |       |22 |  26  | 23|
# 0---+---\--1   |        0---+-8----1   |       0---+-8----1   |
#  \  |    \  \  |         \  17      \  18       \ 17    25 \  18
#   \ |     \  \ |         10 |        12|        10 |  21    12|
#    \|      w  \|           \|         \|          \|         \|
#     4----------5            4----16----5           4----16----5


class FaceType(IntEnum):
    # this indexing is 1 less than the meshfem indexing. Keep that in mind.
    BOTTOM = 0
    RIGHT = 1
    TOP = 2
    LEFT = 3
    FRONT = 4
    BACK = 5

    @staticmethod
    def HEX_27_node_indices_on_type(
        facetype: int,
    ) -> tuple[int, int, int, int, int, int, int, int, int]:
        if facetype == FaceType.BOTTOM:
            return (0, 3, 2, 1, 9, 13, 11, 8, 20)
        if facetype == FaceType.RIGHT:
            return (1, 2, 6, 5, 11, 14, 18, 12, 23)
        if facetype == FaceType.TOP:
            return (4, 5, 6, 7, 16, 18, 19, 17, 25)
        if facetype == FaceType.LEFT:
            return (0, 4, 7, 3, 10, 17, 15, 9, 22)
        if facetype == FaceType.FRONT:
            return (0, 1, 5, 4, 8, 12, 16, 10, 21)
        if facetype == FaceType.BACK:
            return (3, 2, 6, 7, 13, 14, 19, 15, 24)
        msg = f"`facetype` (={facetype}) must be an FaceType"
        raise ValueError(msg)

    @staticmethod
    def HEX_27_edge_to_inds_matrix() -> np.ndarray:
        return np.array([FaceType.HEX_27_node_indices_on_type(i) for i in range(6)])


_QUA_9_facecoords_at_node = np.array(
    [
        (-1, -1),
        (1, -1),
        (1, 1),
        (-1, 1),
        (0, -1),
        (1, 0),
        (0, 1),
        (-1, 0),
        (0, 0),
    ]
)


def QUA_9_facecoords_at_node(node_index: int):
    return _QUA_9_facecoords_at_node[node_index, :]


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
        remapped_a_adj = np.where(
            a.elements_adj >= 0, a_joined_elem_inds[a.elements_adj], -1
        )
        remapped_b_adj = np.where(
            b.elements_adj >= 0, b_joined_elem_inds[b.elements_adj], -1
        )
        nelem = max(np.max(a_joined_elem_inds), np.max(b_joined_elem_inds)) + 1
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
        edges = edges_of_all_elements(element_node_matrix, True).reshape((-1, 3))
        # edges[ielem*4 + edgetype, :] == edgenodes

        # sort edges
        edges_sortinds = np.lexsort(edges.T, axis=0)
        edges_sorted = edges[edges_sortinds, :]

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
        elem_a, edge_a = np.unravel_index(edges_sortinds[paired], (nelem, 4))
        elem_b, edge_b = np.unravel_index(edges_sortinds[paired + 1], (nelem, 4))
        elements_adj[elem_a, edge_a] = elem_b
        elements_adj[elem_b, edge_b] = elem_a

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


def vectorized_bbox_calc(node_coord_matrix: np.ndarray) -> np.ndarray:
    """Computes the bounding boxes for all of the given faces in a fast way.

    Args:
        node_coord_matrix(np.ndarray): ...x9x3 array of node coordinates.
                The last index is the dimension, while the second last is the intra-element node

    Returns:
        np.ndarray: (...x6) array, with bbox = (xmin, ymin, zmin, xmax, ymax, zmax)
    """
    ndim = node_coord_matrix.shape[-1]  # should be 3
    ret = np.empty(
        node_coord_matrix.shape[:-2] + (2 * ndim,), dtype=node_coord_matrix.dtype
    )
    ret_min = ret[..., :ndim]
    ret_max = ret[..., ndim:]

    # initialize (start with corners)
    ret_min[...] = node_coord_matrix[..., 0, :]
    ret_max[...] = node_coord_matrix[..., 0, :]
    for inod in [1, 2, 3]:
        np.minimum(ret_min, node_coord_matrix[..., inod, :], ret_min)
        np.maximum(ret_max, node_coord_matrix[..., inod, :], ret_max)

    # since jacobian is assumed > 1, the only possible critical points are on the boundaries.
    for iedge in range(4):
        edgenodes = EdgeType.QUA_9_node_indices_on_type(iedge)
        edge_coords = node_coord_matrix[..., edgenodes, :]

        # compute critical point
        with np.errstate(divide="ignore", invalid="ignore"):
            crit = np.einsum("k,...kd->...d", maxfind_coefs_b, edge_coords) / np.einsum(
                "k,...kd->...d", maxfind_coefs_a, edge_coords
            )
        valid_search = (crit > -1) * (crit < 1)

        extrema = np.einsum(
            "ji,...i,...j->...",
            L,
            crit[valid_search, None] ** np.arange(3),
            np.swapaxes(edge_coords, -2, -1)[valid_search, :],
        )

        ret_min[valid_search] = np.minimum(ret_min[valid_search], extrema)
        ret_max[valid_search] = np.maximum(ret_max[valid_search], extrema)

    return ret


def unique_edges(
    elements_arr: NDArray[np.uint64], edge_type_arr: NDArray[np.uint8]
) -> tuple[NDArray[np.uint64], NDArray[np.uint8]]:
    """Finds the unique edges among the values passed in.

    In the future, consider modeling this off of np.unique.

    Args:
        elements_arr (NDArray[np.uint64]): element indices
        edge_type_arr (NDArray[np.uint8]): edge types

    Returns:
        tuple[NDArray[np.uint64], NDArray[np.uint8]]: the shortened and sorted unique edges.
    """
    uniques = np.unique(
        np.stack([elements_arr, edge_type_arr], axis=-1, dtype=elements_arr.dtype),
        axis=0,
    )
    return uniques[:, 0], np.astype(uniques[:, 1], np.uint8)
