#include "mesh/IO/fortran/read_material_properties.hpp"
#include "fortranio/interface.hpp"
#include "material/interface.hpp"
#include "specfem_mpi/interface.hpp"
#include "utilities/interface.hpp"
#include <memory>
#include <vector>

struct input_holder {
  // Struct to hold temporary variables read from database file
  type_real val0, val1, val2, val3, val4, val5, val6, val7, val8, val9, val10,
      val11, val12;
  int n, indic;
};

std::vector<std::shared_ptr<specfem::material::material> >
specfem::mesh::IO::fortran::read_material_properties(
    std::ifstream &stream, const int numat, const specfem::MPI::MPI *mpi) {

  input_holder read_values;

  std::vector<std::shared_ptr<specfem::material::material> > materials(numat);

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
      // Acoustic Material
      if (read_values.val2 == 0) {
        const type_real density = read_values.val0;
        const type_real cp = read_values.val1;
        const type_real compaction_grad = read_values.val3;
        const type_real Qkappa = read_values.val5;
        const type_real Qmu = read_values.val6;
        std::shared_ptr<specfem::material::acoustic_material> acoustic_holder =
            std::make_shared<specfem::material::acoustic_material>(
                density, cp, Qkappa, Qmu, compaction_grad);

        materials[read_values.n - 1] = acoustic_holder;
      } else {
        const type_real density = read_values.val0;
        const type_real cp = read_values.val1;
        const type_real cs = read_values.val2;
        const type_real compaction_grad = read_values.val3;
        const type_real Qkappa = read_values.val5;
        const type_real Qmu = read_values.val6;
        std::shared_ptr<specfem::material::elastic_material> elastic_holder =
            std::make_shared<specfem::material::elastic_material>(
                density, cs, cp, Qkappa, Qmu, compaction_grad);
        materials[read_values.n - 1] = elastic_holder;
      }
    } else {
      throw std::runtime_error(
          "Error reading material properties. Invalid material type");
    }
  }

  for (int i = 0; i < materials.size(); i++) {
    mpi->cout(materials[i]->print());
  }

  return materials;
}
