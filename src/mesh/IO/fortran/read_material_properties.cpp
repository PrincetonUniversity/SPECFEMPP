#include "mesh/IO/fortran/read_material_properties.hpp"
#include "fortran_IO.h"
#include "material.h"
#include "specfem_mpi.h"
#include "utils.h"
#include <vector>

std::vector<specfem::material *>
specfem::mesh::IO::fortran::read_material_properties(
    std::ifstream &stream, const int numat, const specfem::MPI::MPI *mpi) {

  specfem::utilities::input_holder read_values;

  std::vector<specfem::material *> materials(numat);

  std::ostringstream message;
  message << "Material systems:\n"
          << "------------------------------";

  mpi->cout(message.str());

  if (mpi->get_rank() == 0)
    std::cout << "Number of material systems = " << numat << "\n\n";

  for (int i = 0; i < numat; i++) {

    specfem::fortran_IO::fortran_read_line(
        stream, &read_values.n, &read_values.indic, &read_values.val0,
        &read_values.val1, &read_values.val2, &read_values.val3,
        &read_values.val4, &read_values.val5, &read_values.val6,
        &read_values.val7, &read_values.val8, &read_values.val9,
        &read_values.val10, &read_values.val11, &read_values.val12);

    if (read_values.n < 1 || read_values.n > numat) {
      throw std::runtime_error("Wrong material set number");
    }

    if (read_values.indic == 1) {
      if (read_values.val2 == 0) {
        specfem::acoustic_material *acoustic_holder =
            new specfem::acoustic_material();
        acoustic_holder->assign(read_values);
        materials[read_values.n - 1] = acoustic_holder;
      } else {
        specfem::elastic_material *elastic_holder =
            new specfem::elastic_material();
        elastic_holder->assign(read_values);
        materials[read_values.n - 1] = elastic_holder;
      }
    } else {
      throw std::runtime_error(
          "Only elastic & acoutsic material has been developed still");
    }
  }

  for (int i = 0; i < materials.size(); i++) {
    mpi->cout(materials[i]->print());
  }

  return materials;
}
