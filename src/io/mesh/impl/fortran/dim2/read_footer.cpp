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

#define UNSUPPORTED_MESHFEM_FOOTER_ERROR                                       \
  {                                                                            \
    throw std::runtime_error(                                                  \
        std::string("Unsupported meshfem footer version: ") +                  \
        std::to_string(version[0]) + "." + std::to_string(version[1]) + "." +  \
        std::to_string(version[2]));                                           \
  }
#define MESHFEM_FOOTER_READ_ERROR(msg)                                         \
  {                                                                            \
    throw std::runtime_error(std::string("Error reading meshfem footer(v") +   \
                             std::to_string(version[0]) + "." +                \
                             std::to_string(version[1]) + "." +                \
                             std::to_string(version[2]) + "): " + msg);        \
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
    UNSUPPORTED_MESHFEM_FOOTER_ERROR;
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
        MESHFEM_FOOTER_READ_ERROR("Unknown footer code " +
                                  std::to_string(sectioncode));
      }
    }

  } else {
    UNSUPPORTED_MESHFEM_FOOTER_ERROR;
  }
}

#undef UNSUPPORTED_MESHFEM_FOOTER_ERROR
#undef MESHFEM_FOOTER_READ_ERROR
