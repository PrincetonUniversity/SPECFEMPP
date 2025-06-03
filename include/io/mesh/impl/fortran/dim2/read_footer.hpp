#pragma once

#include "mesh/mesh.hpp"
#include "specfem_mpi/interface.hpp"
#include <fstream>

namespace specfem {
namespace io {
namespace mesh {
namespace impl {
namespace fortran {
namespace dim2 {
void read_footer(std::ifstream &stream,
                 specfem::mesh::mesh<specfem::dimension::type::dim2> &mesh,
                 const specfem::MPI::MPI *mpi);
}
} // namespace fortran
} // namespace impl
} // namespace mesh
} // namespace io
} // namespace specfem
