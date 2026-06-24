from dataclasses import dataclass, field
from dataclasses import replace as dataclass_replace

import numpy as np
from rtree.index import Index as RTree
from rtree.index import Property as RTreeProperty

from ...helper.index_mapping import IndexMapping
from .faces import FaceType, vectorized_bbox_calc


@dataclass
class BoundarySpec:
    """Keeps track of element indices and faces corresponding to the boundary
    of a volume or collection of volumes.

    One optimization would be to prune (or mark) edges that are part of interfaces.
    """

    element_inds: np.ndarray
    element_faces: np.ndarray
    rtree: RTree = field(default_factory=RTree)
    num_faces: int = field(init=False)

    def __post_init__(self):
        self.num_faces = self.element_inds.size
        # assert self.element_inds.size == self.num_faces
        assert self.element_faces.size == self.num_faces
        assert len(self.rtree) == self.num_faces

    def remapped_elements(self, element_index_mapping: IndexMapping) -> "BoundarySpec":
        return dataclass_replace(
            self, element_inds=element_index_mapping.apply(self.element_inds)
        )

    @staticmethod
    def from_missing_keystones(
        element_nodes: np.ndarray,
        node_coords: np.ndarray,
    ) -> "BoundarySpec":
        """Creates a BoundarySpec object by counting the number of times a central node appears

        If a central node appears twice, then it is internal. If it appears once, it is external.

        Parameters
        ----------
        element_nodes : np.ndarray
            the node tags (shape = (N,27)) of all elements.
        node_coords : np.ndarray
            the coordinates of the nodes, where the indices in element_nodes plug into node_coords.

        Returns
        -------
        BoundarySpec
            The generated BoundarySpec object.
        """
        keystone_node_of_face = FaceType.HEX_27_edge_to_inds_matrix()[:, -1]
        keystone_nodes = element_nodes[:, keystone_node_of_face]

        sorted_keystone_nodes, keystone_inds, keystone_counts = np.unique(
            keystone_nodes, return_index=True, return_counts=True
        )

        externals = keystone_counts == 1

        # keystones MUST appear only once or twice. Something went terribly wrong otherwise
        if not np.all(np.logical_or(externals, keystone_counts == 2)):
            e = RuntimeError("keystone_counts has an invalid value!")
            e.add_note(
                "Keystone occurance counting algorithm to find external faces found a "
                "keystone that appeared more than twice. This should never happen!"
            )
            raise e

        elem_inds, facetypes = np.unravel_index(
            keystone_inds[externals], keystone_nodes.shape
        )

        nfaces = elem_inds.size

        face_bboxes = vectorized_bbox_calc(
            node_coords[
                element_nodes[
                    elem_inds[:, None],
                    FaceType.HEX_27_edge_to_inds_matrix()[facetypes, :],
                ],
                :,
            ]
        )
        # shrink bboxes by a tiny amount
        face_bbox_centers = (face_bboxes[:, :3] + face_bboxes[:, 3:]) / 2
        face_bbox_radii = (face_bboxes[:, 3:] - face_bboxes[:, :3]) * ((1 - 1e-8) / 2)
        face_bboxes[:, :3] = face_bbox_centers - face_bbox_radii
        face_bboxes[:, 3:] = face_bbox_centers + face_bbox_radii

        prop = RTreeProperty()
        prop.dimension = 3

        # https://rtree.readthedocs.io/en/stable/performance.html#use-stream-loading
        def stream_loader():
            for i in range(nfaces):
                yield (i, face_bboxes[i, :], None)

        rtree = RTree(stream_loader(), properties=prop)

        return BoundarySpec(
            element_inds=elem_inds,
            element_faces=facetypes.astype(np.uint8),
            rtree=rtree,
        )
