#include "IO/mesh/impl/fortran/dim3/read_mapping.hpp"
#include "IO/fortranio/interface.hpp"
#include "specfem_setup.hpp"

void specfem::IO::mesh::impl::fortran::dim3::read_ibool(
    std::ifstream &stream,
    specfem::mesh::mapping<specfem::dimension::type::dim3> &mapping,
    const specfem::MPI::MPI *mpi) {

  // Read ibool
  const int nspec = mapping.nspec;

  // Init line reading dummy variable
  std::vector<int> dummy_i(mapping.ngllx * mapping.nglly * mapping.ngllz,
                           -9999);

  int counter = 0;
  // Read element for element
  for (int ispec = 0; ispec < mapping.nspec; ispec++) {
    specfem::IO::fortran_read_line(stream, &dummy_i);
    counter = 0;
    for (int igllz = 0; igllz < mapping.ngllz; igllz++) {
      for (int iglly = 0; iglly < mapping.nglly; iglly++) {
        for (int igllx = 0; igllx < mapping.ngllx; igllx++) {
          mapping.ibool(ispec, igllx, iglly, igllz) = dummy_i[counter] - 1;
          counter++;
        }
      }
    }
  }
  std::cout << "ibool read successfully" << std::endl;

  return;
}
