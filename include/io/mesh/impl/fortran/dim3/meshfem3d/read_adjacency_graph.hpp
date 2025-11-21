#pragma once

#include "enumerations/interface.hpp"
#include "mesh/mesh.hpp"

#include <fstream>

namespace specfem::io::mesh::impl::fortran::dim3::meshfem3d {

specfem::mesh::meshfem3d::adjacency_graph<specfem::dimension::type::dim3>
read_adjacency_graph(std::ifstream &stream, const int nspec);

} // namespace specfem::io::mesh::impl::fortran::dim3::meshfem3d
