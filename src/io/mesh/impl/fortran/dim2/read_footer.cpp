#include "io/mesh/impl/fortran/dim2/read_footer.hpp"
#include "io/fortranio/interface.hpp"
#include <stdexcept>
#include <string>

// .cpp with static functions only; separating files for organization.
#include "footer/adjacency_map.cpp"

static void
read_footer_v0(std::ifstream &stream, const int *version,
               specfem::mesh::mesh<specfem::dimension::type::dim2> &mesh,
               const specfem::MPI::MPI *mpi);

static inline void throw_unsupported_meshfem_footer_error(const int *version) {
  throw std::runtime_error(std::string("Unsupported meshfem footer version: ") +
                           std::to_string(version[0]) + "." +
                           std::to_string(version[1]) + "." +
                           std::to_string(version[2]));
}
static inline void throw_meshfem_footer_read_error(const int *version,
                                                   const std::string &message) {
  throw std::runtime_error(std::string("Error reading meshfem footer(v") +
                           std::to_string(version[0]) + "." +
                           std::to_string(version[1]) + "." +
                           std::to_string(version[2]) + "): " + message);
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

  // footer: use semver (at least until we finalize a format)
  int version[3];
  specfem::io::fortran_read_line(stream, &version[0], &version[1], &version[2]);
  switch (version[0]) {
  case 0:
    read_footer_v0(stream, version, mesh, mpi);
    break;
  default:
    throw_unsupported_meshfem_footer_error(version);
  }
}
#define V0_SECTIONCODE_BREAK (0)
#define V0_SECTIONCODE_ADJACENCYMAP (1)
static void
read_footer_v0(std::ifstream &stream, const int *version,
               specfem::mesh::mesh<specfem::dimension::type::dim2> &mesh,
               const specfem::MPI::MPI *mpi) {
  if (version[1] >= 0) {

    while (true) {
      int sectioncode;
      specfem::io::fortran_read_line(stream, &sectioncode);
      switch (sectioncode) {
      case V0_SECTIONCODE_ADJACENCYMAP:
        footer_0_0_adjmap(stream, mesh, mpi);
        break;
      case V0_SECTIONCODE_BREAK:
        return;
      default:
        throw_meshfem_footer_read_error(
            version, "Unknown footer code " + std::to_string(sectioncode));
      }
    }

  } else {
    throw_unsupported_meshfem_footer_error(version);
  }
}
