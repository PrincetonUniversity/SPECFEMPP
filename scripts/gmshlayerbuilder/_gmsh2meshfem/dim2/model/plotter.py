from typing import Any

import numpy as np

from .edges import EdgeType


def plot_model(node_coordinates: np.ndarray, element_control_nodes: np.ndarray) -> None:
    """Delegated logic for Model.plot() member method.
    """
    import matplotlib.pyplot as plt

    elem_coords = node_coordinates[element_control_nodes, :]

    # plot elem edges
    for edge in EdgeType:
        elem_coords_edge = elem_coords[
            :, EdgeType.QUA_9_node_indices_on_type(edge), :
        ].transpose((1, 0, 2))
        plt.plot(elem_coords_edge[..., 0], elem_coords_edge[..., 2], ".:k", alpha=0.5)
    plt.gca().set_aspect(1)
    plt.show()
