#pragma once

#include "mesh/mesh.hpp"

namespace specfem::io::mesh::impl::fortran::dim3::meshfem3d {

specfem::mesh::meshfem3d::ControlNodes<specfem::dimension::type::dim3>
read_control_nodes(std::ifstream &stream, const specfem::MPI::MPI *mpi);

} // namespace specfem::io::mesh::impl::fortran::dim3::meshfem3d
