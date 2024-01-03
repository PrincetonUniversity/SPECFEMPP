#include "mesh/coupled_interfaces/acoustic_poroelastic.hpp"
#include "fortranio/interface.hpp"

specfem::mesh::coupled_interfaces::acoustic_poroelastic::acoustic_poroelastic(
    const int num_interfaces, std::ifstream &stream,
    const specfem::MPI::MPI *mpi)
    : num_interfaces(num_interfaces),
      acoustic_ispec("acoustic_ispec", num_interfaces),
      poroelastic_ispec("poroelastic_ispec", num_interfaces) {

  if (!num_interfaces)
    return;

  int acoustic_ispec_l, poroelastic_ispec_l;

  for (int i = 0; i < num_interfaces; i++) {
    specfem::fortran_IO::fortran_read_line(stream, &poroelastic_ispec_l,
                                           &acoustic_ispec_l);
    acoustic_ispec(i) = acoustic_ispec_l - 1;
    poroelastic_ispec(i) = poroelastic_ispec_l - 1;
  }

  return;
}
