#include "mesh/coupled_interfaces/elastic_acoustic.hpp"
#include "fortranio/interface.hpp"

specfem::mesh::coupled_interfaces::elastic_acoustic::elastic_acoustic(
    const int num_interfaces, std::ifstream &stream,
    const specfem::MPI::MPI *mpi)
    : num_interfaces(num_interfaces),
      elastic_ispec("elastic_ispec", num_interfaces),
      acoustic_ispec("acoustic_ispec", num_interfaces) {

  if (!num_interfaces)
    return;

  int elastic_ispec_l, acoustic_ispec_l;

  for (int i = 0; i < num_interfaces; i++) {
    specfem::fortran_IO::fortran_read_line(stream, &acoustic_ispec_l,
                                           &elastic_ispec_l);
    elastic_ispec(i) = elastic_ispec_l - 1;
    acoustic_ispec(i) = acoustic_ispec_l - 1;
  }

  return;
}
