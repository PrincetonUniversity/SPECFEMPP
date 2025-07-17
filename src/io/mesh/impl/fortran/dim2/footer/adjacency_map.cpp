#include "mesh/dim2/adjacency_map/adjacency_map.hpp"
#include "enumerations/dimension.hpp"
#include "io/fortranio/interface.hpp"
#include "io/mesh/impl/fortran/dim2/read_footer.hpp"
#include <string>
#include <vector>

/**
 * @brief reads the adjacency map from the database stream and builds the
 * adjacency map struct.
 */
static void
footer_read_adjmap(std::ifstream &stream,
                   specfem::mesh::mesh<specfem::dimension::type::dim2> &mesh,
                   const specfem::MPI::MPI *mpi) {

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
