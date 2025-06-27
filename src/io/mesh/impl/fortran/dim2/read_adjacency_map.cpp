#include "io/mesh/impl/fortran/dim2/read_adjacency_map.hpp"
#include "enumerations/dimension.hpp"
#include "io/fortranio/fortran_io.tpp"
#include "io/fortranio/interface.hpp"
#include "mesh/dim2/adjacency_map/adjacency_map.hpp"
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @brief reads the adjacency map from the database stream and builds the
 * adjacency map struct.
 */

void specfem::io::mesh::impl::fortran::dim2::read_adjacency_map(
    std::ifstream &stream,
    specfem::mesh::mesh<specfem::dimension::type::dim2> &mesh,
    const specfem::MPI::MPI *mpi) {

  bool read_adjacency_map;
  auto current_position = stream.tellg();
  try {
    specfem::io::fortran_read_line(stream, &read_adjacency_map);
  } catch (std::runtime_error &e) {
    stream.clear();
    stream.seekg(current_position);
    read_adjacency_map = false;
  }
  if (!read_adjacency_map) {
    return;
  }

  int n_neighbors;
  std::vector<std::vector<int> > neighbormap(mesh.nspec);

  for (int ispec = 0; ispec < mesh.nspec; ispec++) {
    specfem::io::fortran_read_line(stream, &n_neighbors);
    neighbormap[ispec] = std::vector<int>(n_neighbors);
    auto &adjacencies = neighbormap[ispec];
    specfem::io::fortran_read_line(stream, &adjacencies);
  }

  mesh.adjacency_map = specfem::mesh::adjacency_map::adjacency_map<
      specfem::dimension::type::dim2>(mesh, neighbormap);
}
