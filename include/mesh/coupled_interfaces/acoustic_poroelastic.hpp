#ifndef _COUPLED_ACOUSTIC_POORELASTIC_HPP_
#define _COUPLED_ACOUSTIC_POORELASTIC_HPP_

#include "kokkos_abstractions.h"
#include "specfem_mpi/interface.hpp"

namespace specfem {
namespace mesh {
namespace coupled_interfaces {
struct acoustic_poroelastic {
public:
  acoustic_poroelastic(){};
  acoustic_poroelastic(const int num_interfaces, std::ifstream &stream,
                       const specfem::MPI::MPI *mpi);
  int num_interfaces;
  specfem::kokkos::HostView1d<int> acoustic_ispec;
  specfem::kokkos::HostView1d<int> poroelastic_ispec;
};
} // namespace coupled_interfaces
} // namespace mesh
} // namespace specfem

#endif /* _COUPLED_ACOUSTIC_POORELASTIC_HPP_ */
