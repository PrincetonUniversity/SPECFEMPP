from typing import Any

from _gmshlayerbuilder.dim2.layer_builder.model import Model, EdgeType


try:
    import matplotlib

    mpl_loaded = True
except ImportError:
    matplotlib: Any = None
    mpl_loaded = False


def plot_model(model: Model) -> None:
    if not mpl_loaded:
        msg = (
            "`matplotlib` was not imported. Make sure this python environment "
            "has it installed."
        )
        raise RuntimeError(msg)
    plt = matplotlib.pyplot

    elem_coords = model.nodes[model.elements, :]

    # plot elem edges
    for edge in EdgeType:
        elem_coords_edge = elem_coords[
            :, EdgeType.QUA_9_node_indices_on_type(edge), :
        ].transpose((1, 0, 2))
        plt.plot(elem_coords_edge[..., 0], elem_coords_edge[..., 2], ".:k", alpha=0.5)
    plt.show()
