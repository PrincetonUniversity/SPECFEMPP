from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from ...gmsh_dep import GmshContext
from ...helper import index_mapping
from .model import Model


@dataclass(frozen=True, init=False)
class GmshModel3D:
    """A GmshModel3D wraps a set of volumes defined in gmsh.

    The volumes are handled in both gmsh.model and gmsh.model.mesh spaces.

    The GmshContext must stay active through the lifetime of a GmshModel3D object.
    """

    gmsh: GmshContext
    volumes: list[int]

    def __init__(self, gmsh: GmshContext, volume: int | Iterable[int]):
        """Creates a GmshModel3D object from a single or list of volume entity IDs.

        Parameters
        ----------
        gmsh : GmshContext
            GmshContext object, which must persist through GmshModel3D lifetime
        volume : int | Iterable[int]
            the gmsh.model space entity IDs
        """
        object.__setattr__(self, "gmsh", gmsh)

        object.__setattr__(
            self, "volumes", [volume] if isinstance(volume, int) else list(volume)
        )

    def to_model(self, materials_per_volume: dict[int, int]) -> Model:
        node_tags_per_vol = {}
        node_coords_per_vol = {}
        elements_per_vol = {}

        nelem = 0
        for vol_tag in self.volumes:
            node_tags_per_vol[vol_tag], node_coords_per_vol[vol_tag], _ = (
                self.gmsh.model.mesh.get_nodes(
                    dim=3,
                    tag=vol_tag,
                    includeBoundary=True,
                    returnParametricCoord=False,
                )
            )

            def on_MSH_HEX_27(elems, nodes):
                elements_per_vol[vol_tag] = np.reshape(nodes, (-1, 27))

            self.gmsh.for_element_types_in_entity(3, vol_tag, {12: on_MSH_HEX_27})
            nelem += elements_per_vol[vol_tag].shape[0]

        unique_nodes = np.unique(
            np.concatenate([node_tags_per_vol[vol_tag] for vol_tag in self.volumes])
        )
        nnodes = len(unique_nodes)
        nodes = np.empty((nnodes, 3), dtype=float)
        elements = np.empty((nelem, 27), dtype=int)
        materials = np.empty((nelem,), dtype=int)
        node_reindexing = index_mapping.IndexMapping(unique_nodes)

        element_increment = 0
        for vol_tag in self.volumes:
            node_tags = node_reindexing.apply(node_tags_per_vol[vol_tag])
            nodes[node_tags] = np.reshape(node_coords_per_vol[vol_tag], (-1, 3), "C")
            elems = elements_per_vol[vol_tag]
            element_increment_next = element_increment + elems.shape[0]
            elements[element_increment:element_increment_next] = node_reindexing.apply(
                elems
            )
            materials[element_increment:element_increment_next] = materials_per_volume[
                vol_tag
            ]
            element_increment = element_increment_next

        return Model(nodes=nodes, elements=elements, materials=materials)
