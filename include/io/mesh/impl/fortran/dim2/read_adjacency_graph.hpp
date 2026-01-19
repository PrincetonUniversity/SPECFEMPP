#pragma once

#include "enumerations/interface.hpp"
#include "specfem/mesh.hpp"

namespace specfem::io::mesh::impl::fortran::dim2 {

specfem::mesh::adjacency_graph<specfem::dimension::type::dim2>
read_adjacency_graph(const int nspec, std::ifstream &stream);

}
