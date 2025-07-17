#include "io/mesh/impl/fortran/dim2/read_footer.hpp"
#include "io/fortranio/interface.hpp"
#include <stdexcept>
#include <string>

// .cpp with static functions only; separating files for organization.
#include "footer/adjacency_map.cpp"

static inline void throw_meshfem_footer_read_error(const std::string &message) {
  throw std::runtime_error(std::string("Error reading meshfem footer ") +
                           message);
}

void specfem::io::mesh::impl::fortran::dim2::read_footer(
    std::ifstream &stream,
    specfem::mesh::mesh<specfem::dimension::type::dim2> &mesh,
    const specfem::MPI::MPI *mpi) {

  stream.peek();
  if (stream.eof()) {
    // blank footer. leave mesh unmodified
    return;
  }

  while (true) {
    int sectioncode;
    specfem::io::fortran_read_line(stream, &sectioncode);
    switch (sectioncode) {
    case FOOTERCODE_ADJACENCYMAP:
      footer_read_adjmap(stream, mesh, mpi);
      break;
    case FOOTERCODE_END:
      return;
    default:
      throw_meshfem_footer_read_error("Unknown footer code " +
                                      std::to_string(sectioncode));
    }
  }
}
