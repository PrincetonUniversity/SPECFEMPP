#include "mesh/mesh.hpp"
#include "io/fortranio/interface.hpp"
#include "io/interface.hpp"
#include "io/mesh/impl/fortran/dim3/meshfem3d/read_adjacency_graph.hpp"
#include "io/mesh/impl/fortran/dim3/meshfem3d/read_boundaries.hpp"
#include "io/mesh/impl/fortran/dim3/meshfem3d/read_control_nodes.hpp"
#include "io/mesh/impl/fortran/dim3/meshfem3d/read_materials.hpp"
#include "io/mesh/impl/fortran/dim3/meshfem3d/read_mpi_interfaces.hpp"
#include "io/mesh/impl/fortran/dim3/meshfem3d/read_pml_boundaries.hpp"
#include <fstream>
#include <stdexcept>
#include <string>

specfem::mesh::meshfem3d::mesh<specfem::dimension::type::dim3>
specfem::io::meshfem3d::read_3d_mesh(const std::string &database_file) {

  // Read mesh parameters
  std::ifstream param_stream(database_file, std::ios::in | std::ios::binary);
  if (!param_stream.is_open()) {
    throw std::runtime_error("Could not open mesh parameters file: " +
                             database_file);
  }

  bool mesh_of_earth_chunk;
  specfem::io::fortran_read_line(param_stream, &mesh_of_earth_chunk);

  // TODO (Rohit: EARTH_CHUNK_MESH): Add support for mesh of earth chunk
  if (mesh_of_earth_chunk) {
    throw std::runtime_error(
        "Mesh of earth chunk is not supported in SPECFEM++ yet.");
  }

  specfem::mesh::meshfem3d::mesh<specfem::dimension::type::dim3> mesh;

  mesh.control_nodes =
      specfem::io::mesh::impl::fortran::dim3::meshfem3d::read_control_nodes(
          param_stream);
  const auto [nspec, control_node_index, materials] =
      specfem::io::mesh::impl::fortran::dim3::meshfem3d::read_materials(
          param_stream, mesh.control_nodes.ngnod);
  mesh.nspec = nspec;
  mesh.control_nodes.nspec = nspec;
  mesh.control_nodes.control_node_index = control_node_index;
  mesh.materials = materials;
  mesh.boundaries =
      specfem::io::mesh::impl::fortran::dim3::meshfem3d::read_boundaries(
          param_stream, nspec, mesh.control_nodes);

  // CPML boundaries are not supported yet
  // TODO (Rohit: PML_BOUNDARIES): Add support for PML boundaries
  specfem::io::mesh::impl::fortran::dim3::meshfem3d::read_pml_boundaries(
      param_stream);

  // MPI interfaces are not supported yet
  // TODO (Rohit: MPI_INTERFACES): Add support for MPI interfaces
  specfem::io::mesh::impl::fortran::dim3::meshfem3d::read_mpi_interfaces(
      param_stream);

  // Read adjacency information
  mesh.adjacency_graph =
      specfem::io::mesh::impl::fortran::dim3::meshfem3d::read_adjacency_graph(
          param_stream, mesh.nspec);

  param_stream.close();

  mesh.setup_coupled_interfaces();

  return mesh;
}
