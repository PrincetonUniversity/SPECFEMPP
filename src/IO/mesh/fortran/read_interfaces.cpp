#include "IO/mesh/fortran/read_interfaces.hpp"
#include "interface_container.hpp"
#include "mesh/coupled_interfaces/coupled_interfaces.hpp"
#include "specfem_mpi/interface.hpp"

namespace specfem {
namespace IO {
namespace mesh {
namespace fortran {

  template <specfem::element::medium_tag medium1,
            specfem::element::medium_tag medium2>
  specfem::mesh::interface_container<medium1, medium2>
  read_interfaces(
      const int num_interfaces, std::ifstream &stream, 
      const specfem::MPI::MPI *mpi) {

    if (!num_interfaces)
      return;

    int medium1_ispec_l, medium2_ispec_l;

    specfem::mesh::interface_container<medium1, medium2> interface(num_interfaces);
    interface.medium1_index_mapping("medium1_index_mapping", num_interfaces);
    interface.medium2_index_mapping("medium2_index_mapping", num_interfaces);

    for (int i = 0; i < num_interfaces; i++) {
      specfem::IO::fortran_read_line(stream, &medium2_ispec_l,
                                            &medium1_ispec_l);
      interface.medium1_index_mapping(i) = medium1_ispec_l - 1;
      interface.medium2_index_mapping(i) = medium2_ispec_l - 1;
    }

    return interface;
  }

  specfem::mesh::coupled_interfaces
  read_coupled_interfaces (
      std::ifstream &stream, const int num_interfaces_elastic_acoustic,
      const int num_interfaces_acoustic_poroelastic,
      const int num_interfaces_elastic_poroelastic, const specfem::MPI::MPI *mpi) {

    auto elastic_acoustic = 
    read_interface<
      specfem::element::medium_tag::elastic, 
      specfem::element::medium_tag::acoustic
    >(num_interfaces_elastic_acoustic, stream, mpi);

    auto acoustic_poroelastic = 
    read_interface<
      specfem::element::medium_tag::acoustic, 
      specfem::element::medium_tag::poroelastic
    >(num_interfaces_acoustic_poroelastic, stream, mpi);

    auto elastic_poroelastic = 
    read_interface<
      specfem::element::medium_tag::elastic, 
      specfem::element::medium_tag::poroelastic
    >(num_interfaces_elastic_poroelastic, stream, mpi);
    
    return specfem::mesh::coupled_interfaces(
      elastic_acoustic, acoustic_poroelastic, elastic_poroelastic);
  }



} // namespace fortran
} // namespace mesh
} // namespace IO
} // namespace specfem