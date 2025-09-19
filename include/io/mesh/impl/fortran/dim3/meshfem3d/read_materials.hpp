#pragma once

namespace specfem::io::mesh::impl::fortran::dim3::meshfem3d {

specfem::mesh::meshfem3d::Materials<specfem::dimension::type::dim3>
read_materials(std::ifstream &stream, const specfem::MPI::MPI *mpi);

} // namespace specfem::io::mesh::impl::fortran::dim3::meshfem3d
