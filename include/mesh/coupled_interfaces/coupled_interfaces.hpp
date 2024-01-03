#ifndef _COUPLED_INTERFACES_HPP_
#define _COUPLED_INTERFACES_HPP_

#include "acoustic_poroelastic.hpp"
#include "elastic_acoustic.hpp"
#include "elastic_poroelastic.hpp"
#include "specfem_mpi/interface.hpp"

namespace specfem {
namespace mesh {
namespace coupled_interfaces {

struct coupled_interfaces {
public:
  coupled_interfaces()
      : elastic_acoustic(), acoustic_poroelastic(), elastic_poroelastic(){};
  coupled_interfaces(std::ifstream &stream,
                     const int num_interfaces_elastic_acoustic,
                     const int num_interfaces_acoustic_poroelastic,
                     const int num_interfaces_elastic_poroelastic,
                     const specfem::MPI::MPI *mpi)
      : elastic_acoustic(num_interfaces_elastic_acoustic, stream, mpi),
        acoustic_poroelastic(num_interfaces_acoustic_poroelastic, stream, mpi),
        elastic_poroelastic(num_interfaces_elastic_poroelastic, stream, mpi){};
  specfem::mesh::coupled_interfaces::elastic_acoustic elastic_acoustic;
  specfem::mesh::coupled_interfaces::elastic_poroelastic elastic_poroelastic;
  specfem::mesh::coupled_interfaces::acoustic_poroelastic acoustic_poroelastic;
};

} // namespace coupled_interfaces
} // namespace mesh
} // namespace specfem
#endif /* _COUPLED_INTERFACES_HPP_ */
