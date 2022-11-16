#include "../include/read_material_properties.h"
#include "../include/fortran_IO.h"
#include "../include/material.h"
#include "../include/specfem_mpi.h"
#include "../include/utils.h"
#include <vector>

std::vector<specfem::material>
IO::read_material_properties(std::ifstream &stream, int numat,
                             specfem::MPI *mpi) {

  utilities::value_holder read_values;

  std::vector<specfem::material> materials(numat, specfem::material());

  if (mpi->get_rank() == 0)
    std::cout << "Numat = " << numat << std::endl;

  for (int i = 0; i < numat; i++) {

    IO::fortran_IO::fortran_read_line(
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
        specfem::acoustic_material acoustic_holder;
        acoustic_holder.assign(read_values);
        mpi->cout(acoustic_holder);
        materials[read_values.n - 1] = acoustic_holder;
      } else {
        specfem::elastic_material elastic_holder;
        elastic_holder.assign(read_values);
        mpi->cout(elastic_holder);
        materials[read_values.n - 1] = elastic_holder;
      }
    } else {
      throw std::runtime_error(
          "Only elastic & acoutsic material has been developed still");
    }
  }

  return materials;
}
