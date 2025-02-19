#include "IO/mesh/impl/fortran/read_material_properties.hpp"
#include "IO/fortranio/interface.hpp"
#include "enumerations/interface.hpp"
// #include "mesh/materials/materials.hpp"
#include "mesh/materials/materials.tpp"
#include "specfem_mpi/interface.hpp"
#include "utilities/interface.hpp"
#include <memory>
#include <vector>

// Define some constants for the material properties
constexpr auto elastic_sv = specfem::element::medium_tag::elastic_sv;
constexpr auto isotropic = specfem::element::property_tag::isotropic;
constexpr auto anisotropic = specfem::element::property_tag::anisotropic;
constexpr auto acoustic = specfem::element::medium_tag::acoustic;

struct input_holder {
  // Struct to hold temporary variables read from database file
  type_real val0, val1, val2, val3, val4, val5, val6, val7, val8, val9, val10,
      val11, val12;
  int n, indic;
};

std::vector<specfem::mesh::materials::material_specification> read_materials(
    std::ifstream &stream, const int numat,
    specfem::mesh::materials::material<elastic_sv, isotropic>
        &elastic_isotropic,
    specfem::mesh::materials::material<acoustic, isotropic> &acoustic_isotropic,
    specfem::mesh::materials::material<elastic_sv, anisotropic>
        &elastic_anisotropic,
    const specfem::MPI::MPI *mpi) {

  input_holder read_values;

  std::ostringstream message;

  std::vector<specfem::mesh::materials::material_specification> index_mapping(
      numat);

  message << "Material systems:\n"
          << "------------------------------";

  mpi->cout(message.str());

  if (mpi->get_rank() == 0)
    std::cout << "Number of material systems = " << numat << "\n\n";

  // Section for elastic isotropic
  std::vector<specfem::medium::material<elastic_sv, isotropic> >
      l_elastic_isotropic;

  l_elastic_isotropic.reserve(numat);

  int index_elastic_isotropic = 0;

  // Section for elastic anisotropic
  std::vector<specfem::medium::material<elastic_sv, anisotropic> >
      l_elastic_anisotropic;

  l_elastic_anisotropic.reserve(numat);

  int index_elastic_anisotropic = 0;

  // Section for acoustic isotropic
  std::vector<specfem::medium::material<acoustic, isotropic> >
      l_acoustic_isotropic;

  l_acoustic_isotropic.reserve(numat);

  int index_acoustic_isotropic = 0;

  // Loop over number of materials and read material properties
  for (int i = 0; i < numat; i++) {

    specfem::IO::fortran_read_line(
        stream, &read_values.n, &read_values.indic, &read_values.val0,
        &read_values.val1, &read_values.val2, &read_values.val3,
        &read_values.val4, &read_values.val5, &read_values.val6,
        &read_values.val7, &read_values.val8, &read_values.val9,
        &read_values.val10, &read_values.val11, &read_values.val12);

    if (read_values.n < 1 || read_values.n > numat) {
      throw std::runtime_error(
          "Wrong material set number. Check database file.");
    }

    assert(read_values.n == i + 1);

    // Isotropic material
    if (read_values.indic == 1) {

      // Acoustic Material
      if (read_values.val2 == 0) {
        const type_real density = read_values.val0;
        const type_real cp = read_values.val1;
        const type_real compaction_grad = read_values.val3;
        const type_real Qkappa = read_values.val5;
        const type_real Qmu = read_values.val6;

        specfem::medium::material<acoustic, isotropic>
            acoustic_isotropic_holder(density, cp, Qkappa, Qmu,
                                      compaction_grad);

        acoustic_isotropic_holder.print();

        l_acoustic_isotropic.push_back(acoustic_isotropic_holder);

        index_mapping[i] = specfem::mesh::materials::material_specification(
            specfem::element::medium_tag::acoustic,
            specfem::element::property_tag::isotropic, index_acoustic_isotropic,
            read_values.n - 1);

        index_acoustic_isotropic++;

      } else {

        const type_real density = read_values.val0;
        const type_real cp = read_values.val1;
        const type_real cs = read_values.val2;
        const type_real compaction_grad = read_values.val3;
        const type_real Qkappa = read_values.val5;
        const type_real Qmu = read_values.val6;

        specfem::medium::material<elastic_sv, isotropic>
            elastic_isotropic_holder(density, cs, cp, Qkappa, Qmu,
                                     compaction_grad);

        elastic_isotropic_holder.print();

        l_elastic_isotropic.push_back(elastic_isotropic_holder);

        index_mapping[i] = specfem::mesh::materials::material_specification(
            specfem::element::medium_tag::elastic_sv,
            specfem::element::property_tag::isotropic, index_elastic_isotropic,
            read_values.n - 1);

        index_elastic_isotropic++;
      }
    }
    // Ansotropic material
    else if (read_values.indic == 2) {
      const type_real density = read_values.val0;
      const type_real c11 = read_values.val1;
      const type_real c13 = read_values.val2;
      const type_real c15 = read_values.val3;
      const type_real c33 = read_values.val4;
      const type_real c35 = read_values.val5;
      const type_real c55 = read_values.val6;
      const type_real c12 = read_values.val7;
      const type_real c23 = read_values.val8;
      const type_real c25 = read_values.val9;
      const type_real Qkappa = read_values.val11;
      const type_real Qmu = read_values.val12;

      specfem::medium::material<elastic_sv, anisotropic>
          elastic_anisotropic_holder(density, c11, c13, c15, c33, c35, c55, c12,
                                     c23, c25, Qkappa, Qmu);

      elastic_anisotropic_holder.print();

      l_elastic_anisotropic.push_back(elastic_anisotropic_holder);

      index_mapping[i] = specfem::mesh::materials::material_specification(
          specfem::element::medium_tag::elastic_sv,
          specfem::element::property_tag::anisotropic,
          index_elastic_anisotropic, read_values.n - 1);

      index_elastic_anisotropic++;

    } else {
      throw std::runtime_error("Material type not supported");
    }
  }

  assert(l_elastic_isotropic.size() + l_acoustic_isotropic.size() +
             l_elastic_anisotropic.size() ==
         numat);

  elastic_isotropic = specfem::mesh::materials::material<elastic_sv, isotropic>(
      l_elastic_isotropic.size(), l_elastic_isotropic);

  elastic_anisotropic =
      specfem::mesh::materials::material<elastic_sv, anisotropic>(
          l_elastic_anisotropic.size(), l_elastic_anisotropic);

  acoustic_isotropic = specfem::mesh::materials::material<acoustic, isotropic>(
      l_acoustic_isotropic.size(), l_acoustic_isotropic);

  return index_mapping;
}

void read_material_indices(
    std::ifstream &stream, const int nspec, const int numat,
    const std::vector<specfem::mesh::materials::material_specification>
        &index_mapping,
    const specfem::kokkos::HostView1d<
        specfem::mesh::materials::material_specification>
        material_index_mapping,
    const specfem::kokkos::HostView2d<int> knods,
    const specfem::MPI::MPI *mpi) {

  const int ngnod = knods.extent(0);

  int n, kmato_read, pml_read;

  std::vector<int> knods_read(ngnod, -1);

  for (int ispec = 0; ispec < nspec; ispec++) {
    // format: #element_id  #material_id #node_id1 #node_id2 #...
    specfem::IO::fortran_read_line(stream, &n, &kmato_read, &knods_read,
                                   &pml_read);

    if (n < 1 || n > nspec) {
      throw std::runtime_error("Error reading material indices");
    }

    if (kmato_read < 1 || kmato_read > numat) {
      throw std::runtime_error("Error reading material indices");
    }

    for (int i = 0; i < ngnod; i++) {
      if (knods_read[i] == 0)
        throw std::runtime_error("Error reading knods (node_id) values");

      knods(i, n - 1) = knods_read[i] - 1;
    }

    material_index_mapping(n - 1) = index_mapping[kmato_read - 1];
  }

  return;
}

specfem::mesh::materials
specfem::IO::mesh::impl::fortran::read_material_properties(
    std::ifstream &stream, const int numat, const int nspec,
    const specfem::kokkos::HostView2d<int> knods,
    const specfem::MPI::MPI *mpi) {

  // Create materials instances
  specfem::mesh::materials materials(nspec, numat);

  // Read material properties
  auto index_mapping = read_materials(
      stream, numat, materials.elastic_sv_isotropic,
      materials.acoustic_isotropic, materials.elastic_sv_anisotropic, mpi);

  // Read material indices
  read_material_indices(stream, nspec, numat, index_mapping,
                        materials.material_index_mapping, knods, mpi);

  return materials;
}
