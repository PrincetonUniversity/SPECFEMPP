#include "specfem/io/mesh/impl/fortran/dim3_globe/read_adjacency_graph.hpp"

#include "specfem/io/fortranio/interface.hpp"
#include "specfem/io/mesh/impl/fortran/dim3_globe/globe_codes.hpp"

#include <boost/graph/adjacency_list.hpp>
#include <stdexcept>
#include <vector>

void specfem::io::mesh::impl::fortran::dim3_globe::read_adjacency_graph(
    std::ifstream &stream, specfem::mesh::globe3d_mesh &mesh) {
  using Dimension = specfem::element::dimension_tag;
  using AdjacencyGraph = specfem::mesh::adjacency_graph<Dimension::dim3>;

  int nadjacencies = 0;
  specfem::io::fortran_read_line(stream, &nadjacencies);
  if (nadjacencies < 0) {
    throw std::runtime_error("Invalid adjacency count in globe database");
  }
  std::vector<int> xadj(mesh.nspec + 1), adjncy(nadjacencies),
      adjacency_types(nadjacencies);
  specfem::io::fortran_read_line(stream, &xadj);
  specfem::io::fortran_read_line(stream, &adjncy);
  specfem::io::fortran_read_line(stream, &adjacency_types);

  mesh.adjacency_graph = AdjacencyGraph(mesh.nspec);
  auto &graph = mesh.adjacency_graph.local_connections();
  for (int ispec = 0; ispec < mesh.nspec; ++ispec) {
    if (xadj[ispec] < 1 || xadj[ispec + 1] < xadj[ispec] ||
        xadj[ispec + 1] > nadjacencies + 1) {
      throw std::runtime_error("Invalid CSR adjacency in globe database");
    }
    for (int offset = xadj[ispec] - 1; offset < xadj[ispec + 1] - 1; ++offset) {
      const int neighbor = adjncy[offset] - 1;
      if (neighbor < 0 || neighbor >= mesh.nspec) {
        throw std::runtime_error("Invalid adjacency element in globe database");
      }
      boost::add_edge(
          ispec, neighbor,
          AdjacencyGraph::EdgeProperties(
              specfem::element_connections::type::strongly_conforming,
              specfem::io::mesh::impl::fortran::dim3_globe_impl::to_entity(
                  adjacency_types[offset])),
          graph);
    }
  }
  mesh.adjacency_graph.assert_symmetry();

  int mpi_adjacencies = 0;
  specfem::io::fortran_read_line(stream, &mpi_adjacencies);
  if (mpi_adjacencies < 0) {
    throw std::runtime_error("Invalid MPI adjacency count in globe database");
  }

  auto &mpi_connections = mesh.adjacency_graph.mpi_connections();
  mpi_connections.reserve(mpi_adjacencies);
  for (int i = 0; i < mpi_adjacencies; ++i) {
    int local_element = 0;
    int neighbor_rank = 0;
    int neighbor_element = 0;
    int local_entity = 0;
    int neighbor_entity = 0;
    int local_anchor = 0;
    int neighbor_anchor = 0;
    specfem::io::fortran_read_line(
        stream, &local_element, &neighbor_rank, &neighbor_element,
        &local_entity, &neighbor_entity, &local_anchor, &neighbor_anchor);

    --local_element;
    --neighbor_element;
    // The neighbor index is in the neighbor rank's numbering, but every globe
    // slice has the same nspec, so mesh.nspec bounds it too.
    if (local_element < 0 || local_element >= mesh.nspec ||
        neighbor_element < 0 || neighbor_element >= mesh.nspec ||
        neighbor_rank < 0) {
      throw std::runtime_error("Invalid MPI adjacency in globe database");
    }

    mpi_connections.emplace_back(
        specfem::element_connections::type::strongly_conforming,
        specfem::io::mesh::impl::fortran::dim3_globe_impl::to_entity(
            local_entity),
        static_cast<std::size_t>(neighbor_rank),
        specfem::io::mesh::impl::fortran::dim3_globe_impl::to_entity(
            neighbor_entity),
        static_cast<std::size_t>(local_element),
        static_cast<std::size_t>(neighbor_element),
        specfem::io::mesh::impl::fortran::dim3_globe_impl::to_anchor(
            local_anchor),
        specfem::io::mesh::impl::fortran::dim3_globe_impl::to_anchor(
            neighbor_anchor));
  }
}
