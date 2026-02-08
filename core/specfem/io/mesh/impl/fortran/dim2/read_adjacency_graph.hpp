#pragma once

#include "specfem/enums.hpp"
#include "specfem/mesh.hpp"

namespace specfem::io::mesh::impl::fortran::dim2 {

specfem::mesh::adjacency_graph<specfem::element::dimension_tag::dim2>
read_adjacency_graph(const int nspec, std::ifstream &stream);

}
