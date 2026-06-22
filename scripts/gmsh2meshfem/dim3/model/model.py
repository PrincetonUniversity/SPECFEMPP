from dataclasses import dataclass, field
from dataclasses import replace as dataclass_replace
from typing import Iterable

import numpy as np

from ...gmsh_dep import GmshContext

# from .boundary import BoundarySpec
# from .edges import ConformingInterfaces
from ...helper.index_mapping import IndexMapping, JoinedIndexMapping
from .boundary import BoundarySpec
from .nonconforming_interfaces import NonconformingInterfaces

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
    nodes: np.ndarray  # float(nnodes, 3)
    elements: np.ndarray  # int(nelem, ngnod = 8|27)
    materials: np.ndarray  # int(nelem,)

    boundary_faces: BoundarySpec = field(init=False)
    nonconforming_interfaces: NonconformingInterfaces = field(init=False)

    def __post_init__(self):
        self.boundary_faces = BoundarySpec.from_missing_keystones(
            self.elements, self.nodes
        )
        self.nonconforming_interfaces = NonconformingInterfaces.from_boundaryspec(
            bdspec=self.boundary_faces,
            node_locs=self.nodes,
            element_nodes=self.elements,
        )

    @property
    def num_nodes(self):
        return self.nodes.shape[0]

    @property
    def num_elements(self):
        return self.elements.shape[0]
