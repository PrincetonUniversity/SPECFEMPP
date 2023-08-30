#ifndef _COUPLED_ELASTIC_POORELASTIC_HPP_
#define _COUPLED_ELASTIC_POORELASTIC_HPP_

#include "kokkos_abstractions.h"
#include "specfem_mpi/interface.hpp"

namespace specfem {
namespace mesh {
namespace coupled_interfaces {
struct elastic_poroelastic {
public:
  elastic_poroelastic(){};
  elastic_poroelastic(const int num_interfaces, std::ifstream &stream,
                      const specfem::MPI::MPI *mpi);
  int num_interfaces = 0;
  specfem::kokkos::HostView1d<int> elastic_ispec;
  specfem::kokkos::HostView1d<int> poroelastic_ispec;
};
} // namespace coupled_interfaces
} // namespace mesh
} // namespace specfem

#endif /* _COUPLED_ELASTIC_POORELASTIC_HPP_ */
