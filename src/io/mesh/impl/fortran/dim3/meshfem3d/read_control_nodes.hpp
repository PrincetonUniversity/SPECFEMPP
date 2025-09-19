
#include "io/mesh/impl/fortran/dim3/meshfem3d/read_control_nodes.hpp"
#include "mesh/mesh.hpp"

specfem::mesh::meshfem3d::ControlNodes<specfem::dimension::type::dim3>
specfem::io::mesh::impl::fortran::dim3::meshfem3d::read_control_nodes(
    const std::ifstream &stream, const specfem::MPI::MPI *mpi) {

  using ControlNodesType =
      specfem::mesh::meshfem3d::ControlNodes<specfem::dimension::type::dim3>;

  int nnodes;
  specfem::io::read_fortran_line(stream, &nnodes);
  ControlNodesType control_nodes(nnodes);

  // Read control nodes coordinates one by one
  for (int inode = 0; inode < nnodes; ++inode) {
    int index;
    type_real x, y, z;
    specfem::io::read_fortran_line(stream, &index, &x, &y, &z);
    control_nodes.coordinates(inode, 0) = x;
    control_nodes.coordinates(inode, 1) = y;
    control_nodes.coordinates(inode, 2) = z;
  }

  return control_nodes;
}
