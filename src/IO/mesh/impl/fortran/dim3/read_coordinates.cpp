#include "IO/mesh/impl/fortran/dim3/read_coordinates.hpp"
#include "specfem_setup.hpp"

void specfem::IO::mesh::impl::fortran::dim3::read_xyz(
    std::ifstream &stream,
    specfem::mesh::coordinates<specfem::dimension::type::dim3> &coordinates,
    const specfem::MPI::MPI *mpi) {

  std::vector<type_real> dummy_f(coordinates.nglob, -9999.0);

  // Read xstore_unique
  specfem::IO::fortran_read_line(stream, &dummy_f);
  for (int i = 0; i < coordinates.nglob; i++) {
    coordinates.x(i) = dummy_f[i];
  }

  // Read ystore_unique
  specfem::IO::fortran_read_line(stream, &dummy_f);
  for (int i = 0; i < coordinates.nglob; i++) {
    coordinates.y(i) = dummy_f[i];
  }

  // Read zstore_unique
  specfem::IO::fortran_read_line(stream, &dummy_f);
  std::cout << "Read z" << std::endl;
  for (int i = 0; i < coordinates.nglob; i++) {
    coordinates.z(i) = dummy_f[i];
  }
}
