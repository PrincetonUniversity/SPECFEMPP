from dataclasses import dataclass, field

import numpy as np
from _gmshlayerbuilder.gmsh_dep import GmshContext

from .edges import EdgeType


@dataclass
class BoundarySpec:
    """Keeps track of element indices and edges corresponding to the boundary of
    a surface or collection of surfaces.
    """

    @dataclass(frozen=True)
    class EntityKey:
        entity_tag: int
        start_index: int
        end_index: int

        @property
        def num_edges_in_entity(self):
            return self.end_index - self.start_index

        def _edges_in_entity(
            self, parent: "BoundarySpec"
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            return (
                parent.edge_tags[self.start_index : self.end_index],
                parent.element_inds[self.start_index : self.end_index],
                parent.element_edges[self.start_index : self.end_index],
            )

    edge_tags: np.ndarray
    element_inds: np.ndarray
    element_edges: np.ndarray
    boundary_entity_spec: dict[int, EntityKey]
    num_edges: int = field(init=False)

    def __post_init__(self):
        self.num_edges = self.edge_tags.size
        assert self.element_inds.size == self.num_edges
        assert self.element_edges.size == self.num_edges

    @staticmethod
    def from_model_entity(
        gmsh: GmshContext,
        edge_entity: list[int] | int,
        element_nodes: np.ndarray,
    ) -> "BoundarySpec":
        """For a model edge (entity) or list of model edges,
        recovers the pairs `(ielem, edgetype)` for each mesh edge in the
        model edge or list of edges.
        `ielem` is the index of the element, remapped by element_mapping.
        `element_nodes` are the un-remapped node indices of each element,
        in the same order as `element_mapping.original_tag_list`.

        Args:
            gmsh (GmshContext): The gmsh context to use
            edge_entity (list[int] | int): model entity tag(s).
            element_mapping (IndexMapping): the index mapping used for the elements
            element_nodes (np.ndarray): the node tags (shape = (N,9)) of all elements.
        """
        if isinstance(edge_entity, int):
            edge_entity = [edge_entity]

        # for each model (entity) edge tag, (mesh edge tags, elem_indices, elem_edgetypes)
        collected_values: dict[
            int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]
        ] = {}
        num_collected_edges = 0

        QUA_9_node_indices_on_type = np.array(
            [EdgeType.QUA_9_node_indices_on_type(i) for i in range(4)]
        )

        for model_edge_tag in edge_entity:
            if model_edge_tag not in collected_values:
                collected_values[model_edge_tag] = []

            def on_MSH_LIN_2(elems, nodes):
                msg = (
                    "At the moment, 2-node lines have not been implemented. "
                    "Please mesh at order-2 by setting "
                    '`gmsh.option.setNumber("Mesh.ElementOrder", 2)`.'
                )
                raise NotImplementedError(msg)

            def on_MSH_LIN_3(edge_elems, edge_nodes):
                nonlocal num_collected_edges
                max_vec_size = 64  # we will only vectorize up to k mesh edges at a time
                edge_nodes = edge_nodes.reshape((-1, 3))
                n_edges = edge_elems.size

                edge_counts = np.zeros(n_edges, np.int32)
                elem_inds = np.empty(n_edges, np.int64)
                elem_edges = np.empty(n_edges, np.uint8)

                for istart in range(0, n_edges, max_vec_size):
                    iend = min(istart + max_vec_size, n_edges)
                    candidate_elem, edge_for_candidate = np.nonzero(
                        # [ielem,iedge]: iedge has all nodes in ielem?
                        np.all(
                            # [ielem,iedge,edgenode_ind]: (iedge,edgenode_ind) in element?
                            np.any(
                                # [ielem,iedge,edgenode_ind,elem_node_ind]
                                element_nodes[:, None, None, :-1]
                                == edge_nodes[None, istart:iend, :, None],
                                axis=-1,
                            ),
                            axis=-1,
                        )
                    )
                    edge_for_candidate += istart
                    # for candidate: which elem_node_inds are hit?
                    candidate_flags = np.any(
                        element_nodes[candidate_elem, :, None]
                        == edge_nodes[edge_for_candidate, None, :],
                        axis=-1,
                    )

                    # all elem_node_inds of edge hit? [candidate_ind, edgetype]
                    candidate_edgetypes = np.all(
                        candidate_flags[:, QUA_9_node_indices_on_type], axis=-1
                    )

                    successful_candidates, successful_edgetypes = np.nonzero(
                        candidate_edgetypes
                    )
                    successful_edges = edge_for_candidate[successful_candidates]

                    # collapse by collecting one entry per edge and
                    # count number of times edge was found
                    unique_edges, unique_success_inds, unique_counts = np.unique(
                        successful_edges,
                        return_inverse=False,
                        return_index=True,
                        return_counts=True,
                    )
                    edge_counts[unique_edges] += unique_counts
                    elem_inds[unique_edges] = candidate_elem[
                        successful_candidates[unique_success_inds]
                    ]
                    elem_edges[unique_edges] = successful_edgetypes

                # ensure all edges were hit at least once, and not more than twice.
                if np.any(edge_counts == 0):
                    rte = RuntimeError(
                        "Error in computing BoundarySpec of model edge with tag "
                        f"{model_edge_tag}. At least one mesh edge does not have a mesh "
                        "element attached."
                    )
                    (failtags,) = np.nonzero(edge_counts == 0)
                    rte.add_note(
                        f"mesh tags with no found elements:\n{edge_elems[failtags]}"
                    )
                    raise rte

                if np.any(edge_counts > 2):
                    rte = RuntimeError(
                        "Error in computing BoundarySpec of model edge with tag "
                        f"{model_edge_tag}. At least one mesh edge has more than two mesh "
                        "element attached. This should never happen"
                    )
                    (failtags,) = np.nonzero(edge_counts > 2)
                    rte.add_note(
                        f"mesh tags with 3+ found elements:\n{edge_elems[failtags]}"
                    )
                    raise rte

                (edges_of_one,) = np.nonzero(edge_counts == 1)
                collected_values[model_edge_tag].append(
                    (
                        edge_elems[edges_of_one],
                        elem_inds[edges_of_one],
                        elem_edges[edges_of_one],
                    )
                )
                num_collected_edges += edges_of_one.size

            gmsh.for_element_types_in_entity(
                1,
                model_edge_tag,
                {
                    1: on_MSH_LIN_2,
                    8: on_MSH_LIN_3,
                },
            )

        # collect by model (entity) tag
        edge_tags = np.empty(num_collected_edges, dtype=np.int64)
        element_inds = np.empty(num_collected_edges, dtype=np.int64)
        element_edges = np.empty(num_collected_edges, dtype=np.uint8)

        # recollect into ^
        num_collected_edges = 0
        entity_spec = {}
        for model_edge_tag, val_list in collected_values.items():
            start = num_collected_edges
            for tag, elem, edge in val_list:
                val_end = num_collected_edges + tag.size
                edge_tags[num_collected_edges:val_end] = tag
                element_inds[num_collected_edges:val_end] = elem
                element_edges[num_collected_edges:val_end] = edge

                num_collected_edges = val_end

            entity_spec[model_edge_tag] = BoundarySpec.EntityKey(
                entity_tag=model_edge_tag,
                start_index=start,
                end_index=num_collected_edges,
            )

        return BoundarySpec(
            edge_tags=edge_tags,
            element_inds=element_inds,
            element_edges=element_edges,
            boundary_entity_spec=entity_spec,
        )

    @staticmethod
    def union(bdspec1: "BoundarySpec", bdspec2: "BoundarySpec") -> "BoundarySpec":
        num_edges = bdspec1.num_edges + bdspec2.num_edges
        edge_tags = np.empty(num_edges, dtype=np.int64)
        element_inds = np.empty(num_edges, dtype=np.int64)
        element_edges = np.empty(num_edges, dtype=np.uint8)
        edges_start = 0
        entity_spec = {}
        for entity_ind in set(bdspec1.boundary_entity_spec.keys()) | set(
            bdspec2.boundary_entity_spec.keys()
        ):
            if entity_ind in bdspec1.boundary_entity_spec:
                spec1 = bdspec1.boundary_entity_spec[entity_ind]
                if entity_ind in bdspec2.boundary_entity_spec:
                    spec2 = bdspec2.boundary_entity_spec[entity_ind]
                    edges_mid = edges_start + spec1.num_edges_in_entity
                    edges_end = edges_mid + spec2.num_edges_in_entity
                    entity_spec[entity_ind] = BoundarySpec.EntityKey(
                        entity_tag=entity_ind,
                        start_index=edges_start,
                        end_index=edges_end,
                    )
                    (
                        edge_tags[edges_start:edges_mid],
                        element_inds[edges_start:edges_mid],
                        element_edges[edges_start:edges_mid],
                    ) = spec1._edges_in_entity(bdspec1)
                    (
                        edge_tags[edges_mid:edges_end],
                        element_inds[edges_mid:edges_end],
                        element_edges[edges_mid:edges_end],
                    ) = spec2._edges_in_entity(bdspec2)
                else:
                    edges_end = edges_start + spec1.num_edges_in_entity
                    entity_spec[entity_ind] = BoundarySpec.EntityKey(
                        entity_tag=entity_ind,
                        start_index=edges_start,
                        end_index=edges_end,
                    )
                    (
                        edge_tags[edges_start:edges_end],
                        element_inds[edges_start:edges_end],
                        element_edges[edges_start:edges_end],
                    ) = spec1._edges_in_entity(bdspec1)
            else:
                spec2 = bdspec2.boundary_entity_spec[entity_ind]
                edges_end = edges_start + spec2.num_edges_in_entity
                entity_spec[entity_ind] = BoundarySpec.EntityKey(
                    entity_tag=entity_ind,
                    start_index=edges_start,
                    end_index=edges_end,
                )
                (
                    edge_tags[edges_start:edges_end],
                    element_inds[edges_start:edges_end],
                    element_edges[edges_start:edges_end],
                ) = spec2._edges_in_entity(bdspec2)
            edges_start = edges_end

        return BoundarySpec(
            edge_tags=edge_tags,
            element_inds=element_inds,
            element_edges=element_edges,
            boundary_entity_spec=entity_spec,
        )
