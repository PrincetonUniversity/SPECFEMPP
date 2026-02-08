#pragma once

#include "specfem/enums.hpp"
#include "specfem/mesh.hpp"

#include <fstream>

namespace specfem::io::mesh::impl::fortran::dim3 {

specfem::mesh::adjacency_graph<specfem::element::dimension_tag::dim3>
read_adjacency_graph(std::ifstream &stream, const int nspec);

} // namespace specfem::io::mesh::impl::fortran::dim3
