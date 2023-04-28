#include "kokkos_abstractions.h"
#include "source/interface.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include <cmath>

void specfem::sources::source::check_locations(const type_real xmin,
                                               const type_real xmax,
                                               const type_real zmin,
                                               const type_real zmax,
                                               const specfem::MPI::MPI *mpi) {
  specfem::utilities::check_locations(this->get_x(), this->get_z(), xmin, xmax,
                                      zmin, zmax, mpi);
}

std::ostream &
specfem::sources::operator<<(std::ostream &out,
                             const specfem::sources::source &source) {
  source.print(out);
  return out;
}
