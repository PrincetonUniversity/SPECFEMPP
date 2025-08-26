import matplotlib.pyplot as plt
import numpy as np

from .layer_builder.layer_boundaries import LerpLayerBoundary
from .layer_builder.layer import Layer
from .layer_builder.layeredbuilder import LayeredBuilder
from .layer_builder.model import EdgeType

from .exporter import Exporter2D

def demo():
    builder = LayeredBuilder(-50, 50)
    layer = LerpLayerBoundary()
    layer.points = [(-50, -20), (50, -20)]
    builder.boundaries.append(layer)
    layer = LerpLayerBoundary()
    layer.points = [(-50, 0), (-20, 5), (0, 10), (20, -3), (50, 0)]
    builder.boundaries.append(layer)
    builder.layers = [Layer(10, 3)]


    #two layers
    layer = LerpLayerBoundary()
    layer.points = [(-50, 35), (-40, 30), (5, 29), (20, 30), (50, 35)]
    builder.boundaries.append(layer)
    builder.layers = [Layer(10, 3), Layer(20, 5)]

    #three layers
    layer = LerpLayerBoundary()
    layer.points = [(-50, 40), (50, 40)]
    builder.boundaries.append(layer)
    builder.layers = [Layer(10, 3), Layer(20, 5), Layer(30, 5)]

    model = builder.create_model()

    elems = model.nodes[model.elements, :]
    midpoints = np.mean(elems, axis=1)
    c = 0.5
    pts_plt = (1 - c) * elems + c * midpoints[:, None, :]
    plt.scatter(
        pts_plt[..., 0],
        pts_plt[..., 2],
        c=np.broadcast_to(model.materials[:, None], (midpoints.shape[0], 9)),
    )

    QUA_9_node_indices_on_type = np.array(
        [EdgeType.QUA_9_node_indices_on_type(i) for i in range(4)]
    )
    edge_to_keynode = QUA_9_node_indices_on_type[:, 1]

    for bd_entity in model.boundaries.boundary_entity_spec.values():
        marked_node_inds = model.elements[
            model.boundaries.element_inds[
                bd_entity.start_index : bd_entity.end_index, None
            ],
            QUA_9_node_indices_on_type[
                model.boundaries.element_edges[
                    bd_entity.start_index : bd_entity.end_index
                ]
            ],
        ]
        plt.scatter(
            model.nodes[marked_node_inds, 0],
            model.nodes[marked_node_inds, 2],
            label=f"boundary elem locations (entity {bd_entity.entity_tag})",
            marker="x",
        )
        bd_entity.entity_tag

    # draw conforming interfaces
    coupled_elem, coupled_edge = np.where(
        model.conforming_interfaces.elements_adj != -1
    )
    draw_pts = np.empty((2, coupled_elem.size, 3))
    draw_pts[0, :, :] = pts_plt[coupled_elem, edge_to_keynode[coupled_edge], :]
    draw_pts[1, :, :] = midpoints[
        model.conforming_interfaces.elements_adj[coupled_elem, coupled_edge], :
    ]
    plt.plot(draw_pts[..., 0], draw_pts[..., 2], ":k")

    # plt.scatter(
    #     model.nodes[model.top_nodes, 0],
    #     model.nodes[model.top_nodes, 2],
    #     label="top node original locations",
    #     marker="x",
    # )
    # plt.scatter(
    #     model.nodes[model.bottom_nodes, 0],
    #     model.nodes[model.bottom_nodes, 2],
    #     label="bottom node original locations",
    #     marker="x",
    # )
    # plt.scatter(
    #     model.nodes[model.left_nodes, 0],
    #     model.nodes[model.left_nodes, 2],
    #     label="left node original locations",
    #     marker="+",
    # )
    # plt.scatter(
    #     model.nodes[model.right_nodes, 0],
    #     model.nodes[model.right_nodes, 2],
    #     label="right node original locations",
    #     marker="+",
    # )

    # draw lines between linked nodes
    join_nodes_a = pts_plt[
        model.nonconforming_interfaces.elements_a,
        edge_to_keynode[model.nonconforming_interfaces.edges_a],
        :,
    ]
    join_nodes_b = pts_plt[
        model.nonconforming_interfaces.elements_b,
        edge_to_keynode[model.nonconforming_interfaces.edges_b],
        :,
    ]
    join_locs = np.stack([join_nodes_a, join_nodes_b], axis=0)
    if join_locs.shape[1] > 0:
        plt.plot(join_locs[:, 1:, 0], join_locs[:, 1:, 2], "k")
        plt.plot(
            join_locs[:, 0, 0],
            join_locs[:, 0, 2],
            "k",
            label="nonconforming interfaces",
        )

    plt.title(f"Multilayer Bathemetry (elements scaled {c:.0%} about centroids)")

    plt.legend()
    plt.show()

    Exporter2D(model,"./params",nonconforming_adjacencies_file="nc_adj").export_mesh()
