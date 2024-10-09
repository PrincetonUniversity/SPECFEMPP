
#include "interface_container.hpp"
#include "specfem_mpi/interface.hpp"

namespace specfem {
namespace IO {
namespace mesh {
namespace fortran {

template <specfem::element::medium_tag medium1,
          specfem::element::medium_tag medium2>
specfem::mesh::interface_container<medium1, medium2>
read_interface(
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




} // namespace fortran
} // namespace mesh
} // namespace IO
} // namespace specfem