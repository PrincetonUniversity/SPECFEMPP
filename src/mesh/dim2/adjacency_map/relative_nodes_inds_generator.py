import pathlib

edges = ["RIGHT", "TOP", "LEFT", "BOTTOM"]

# if counterclockwise direction is in the + direction
CCW_is_positive = [True, False, False, True]
# the along-edge direction (x varies on this edge)
edge_axis_is_x = [False, True, False, True]
# the constant index is zero or ngll
is_edge_at_zero = [False, False, True, True]


def should_flip(edge1: int, edge2: int):
    """Returns whether or not the along-edge coordinate is flipped when edge1 and edge2
    are conformally joined.
    """
    return CCW_is_positive[edge1] == CCW_is_positive[edge2]


funcstr = """
#include "mesh/dim2/adjacency_map/adjacency_map.hpp"

static inline void relative_node_inds(int &ix_adj, int &iz_adj, const int iedge,
                                      const int edgeind, const int edgeind_adj,
                                      const int ngll) {
  switch (edgeind) {
%s  }
}
"""


spacing = "  "


def macro_EDGEIND_OF(edge, spacing_size):
    return (
        "specfem::mesh::adjacency_map::\n"
        + (spacing * spacing_size)
        + "adjacency_map<specfem::dimension::type::dim2>"
        + "::edge_to_index(\n"
        + (spacing * (spacing_size + 2))
        + f"specfem::enums::edge::type::{edge})"
    )


internal = ""
for edge1, estr1 in enumerate(edges):
    internal += spacing + f"case {macro_EDGEIND_OF(estr1, 3)}:\n"
    internal += spacing * 2 + "switch (edgeind_adj) {\n"
    for edge2, estr2 in enumerate(edges):
        fixed_axis = "iz_adj" if edge_axis_is_x[edge2] else "ix_adj"
        fixed_coord = "0" if is_edge_at_zero[edge2] else "ngll - 1"
        vary_axis = "ix_adj" if edge_axis_is_x[edge2] else "iz_adj"
        vary_coord = "ngll - 1 - iedge" if should_flip(edge1, edge2) else "iedge"
        internal += spacing * 2 + f"case {macro_EDGEIND_OF(estr2, 4)}:\n"
        internal += spacing * 3 + f"{fixed_axis} = {fixed_coord};\n"
        internal += spacing * 3 + f"{vary_axis} = {vary_coord};\n"
        internal += spacing * 3 + "return;\n"
    internal += spacing * 2 + "}\n"

print(funcstr % internal)
with (pathlib.Path(__file__).parent / "relative_node_inds.cpp").open("w") as f:
    f.write(funcstr % internal)
