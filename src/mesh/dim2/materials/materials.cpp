#include "mesh/dim2/materials/materials.hpp"
#include "IO/fortranio/interface.hpp"
#include "kokkos_abstractions.h"
#include "mesh/dim2/materials/materials.tpp"
#include <vector>

std::variant<
    specfem::medium::material<specfem::element::medium_tag::elastic,
                              specfem::element::property_tag::isotropic>,
    specfem::medium::material<specfem::element::medium_tag::elastic,
                              specfem::element::property_tag::anisotropic>,
    specfem::medium::material<specfem::element::medium_tag::acoustic,
                              specfem::element::property_tag::isotropic> >
specfem::mesh::materials::operator[](const int index) const {

  const auto &material_specification = this->material_index_mapping(index);

  // Return the material properties based on the material specification

  // Return elastic isotropic
  if (material_specification.type == specfem::element::medium_tag::elastic &&
      material_specification.property ==
          specfem::element::property_tag::isotropic) {
    return this->elastic_isotropic
        .material_properties[material_specification.index];

    // Return elastic anisotropic
  } else if (material_specification.type ==
                 specfem::element::medium_tag::elastic &&
             material_specification.property ==
                 specfem::element::property_tag::anisotropic) {
    return this->elastic_anisotropic
        .material_properties[material_specification.index];

    // Return acoustic isotropic
  } else if (material_specification.type ==
                 specfem::element::medium_tag::acoustic &&
             material_specification.property ==
                 specfem::element::property_tag::isotropic) {
    return this->acoustic_isotropic
        .material_properties[material_specification.index];
    // Throw an error if the material type is not supported
  } else {
    throw std::runtime_error("Material type not supported");
  }

  return {};
}

// specfem::mesh::material_ind::material_ind(
//     std::ifstream &stream, const int ngnod, const int nspec, const int numat,
//     const specfem::kokkos::HostView2d<int> knods,
//     const specfem::MPI::MPI *mpi) {
//   std::vector<int> knods_read(ngnod, -1);
//   int n, kmato_read, pml_read;

//   // Allocate views
//   *this = specfem::mesh::material_ind(nspec, ngnod);

//   // Read an assign material values, coordinate numbering, PML association
//   for (int ispec = 0; ispec < nspec; ispec++) {
//     // format: #element_id  #material_id #node_id1 #node_id2 #...
//     specfem::IO::fortran_read_line(stream, &n, &kmato_read,
//     &knods_read,
//                                            &pml_read);

//     // material association
//     if (n < 1 || n > nspec) {
//       throw std::runtime_error("Error reading mato properties");
//     }
//     this->kmato(n - 1) = kmato_read - 1;
//     this->region_CPML(n - 1) = pml_read;

//     // element control node indices (ipgeo)
//     for (int i = 0; i < ngnod; i++) {
//       if (knods_read[i] == 0)
//         throw std::runtime_error("Error reading knods (node_id) values");

//       knods(i, n - 1) = knods_read[i] - 1;
//     }
//   }

//   for (int ispec = 0; ispec < nspec; ispec++) {
//     int imat = this->kmato(ispec);
//     if (imat < 0 || imat >= numat) {
//       throw std::runtime_error(
//           "Error reading material properties. Invalid material ID number");
//     }
//   }
// }
