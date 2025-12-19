#include "io/mesh/impl/fortran/dim2/read_material_properties.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/interface.hpp"
#include "io/fortranio/interface.hpp"
#include "mesh/mesh.hpp"
#include "specfem/logger.hpp"
#include "specfem_mpi/interface.hpp"
#include "utilities/interface.hpp"
#include <memory>
#include <sstream>
#include <vector>

// Define some constants for the material properties
constexpr auto acoustic = specfem::element::medium_tag::acoustic;
constexpr auto elastic_psv = specfem::element::medium_tag::elastic_psv;
constexpr auto elastic_sh = specfem::element::medium_tag::elastic_sh;
constexpr auto elastic_psv_t = specfem::element::medium_tag::elastic_psv_t;
constexpr auto electromagnetic_te =
    specfem::element::medium_tag::electromagnetic_te;
constexpr auto poroelastic = specfem::element::medium_tag::poroelastic;
constexpr auto isotropic = specfem::element::property_tag::isotropic;
constexpr auto isotropic_cosserat =
    specfem::element::property_tag::isotropic_cosserat;
constexpr auto anisotropic = specfem::element::property_tag::anisotropic;

struct input_holder {
  // Struct to hold temporary variables read from database file
  double val0, val1, val2, val3, val4, val5, val6, val7, val8, val9, val10,
      val11, val12;
  int n, indic;
};

std::vector<specfem::mesh::materials<
    specfem::dimension::type::dim2>::material_specification>
read_materials(
    std::ifstream &stream, const int numat,
    const specfem::enums::elastic_wave elastic_wave,
    const specfem::enums::electromagnetic_wave electromagnetic_wave,
    specfem::mesh::materials<specfem::dimension::type::dim2>::material<
        acoustic, isotropic> &acoustic_isotropic,
    specfem::mesh::materials<specfem::dimension::type::dim2>::material<
        elastic_psv, isotropic> &elastic_psv_isotropic,
    specfem::mesh::materials<specfem::dimension::type::dim2>::material<
        elastic_sh, isotropic> &elastic_sh_isotropic,
    specfem::mesh::materials<specfem::dimension::type::dim2>::material<
        elastic_psv, anisotropic> &elastic_psv_anisotropic,
    specfem::mesh::materials<specfem::dimension::type::dim2>::material<
        elastic_sh, anisotropic> &elastic_sh_anisotropic,
    specfem::mesh::materials<specfem::dimension::type::dim2>::material<
        poroelastic, isotropic> &poroelastic_isotropic,
    specfem::mesh::materials<specfem::dimension::type::dim2>::material<
        electromagnetic_te, isotropic> &electromagnetic_te_isotropic,
    specfem::mesh::materials<specfem::dimension::type::dim2>::material<
        elastic_psv_t, isotropic_cosserat> &elastic_psv_t_isotropic_cosserat,
    const specfem::MPI::MPI *mpi) {

  // Define the elastic medium tag based on input elastic wave type
  const specfem::element::medium_tag elastic = [elastic_wave]() {
    if (elastic_wave == specfem::enums::elastic_wave::psv) {
      return specfem::element::medium_tag::elastic_psv;
    } else if (elastic_wave == specfem::enums::elastic_wave::sh) {
      return specfem::element::medium_tag::elastic_sh;
    } else {
      std::ostringstream message;
      message << "Elastic wave type not supported for elastic material ["
              << __FILE__ << ":" << __LINE__ << "]\n";
      throw std::runtime_error(message.str());
    }
  }();

  const specfem::element::medium_tag electromagnetic =
      [electromagnetic_wave]() {
        if (electromagnetic_wave == specfem::enums::electromagnetic_wave::te) {
          return specfem::element::medium_tag::electromagnetic_te;
        } else {
          std::ostringstream message;
          message
              << "TM wave type not yet supported for electromagnetic material ["
              << __FILE__ << ":" << __LINE__ << "]\n";
          throw std::runtime_error(message.str());
        }
      }();

  input_holder read_values;

  std::ostringstream message;

  std::vector<specfem::mesh::materials<
      specfem::dimension::type::dim2>::material_specification>
      index_mapping(numat);

  message << "Material systems:\n"
          << "------------------------------";

  specfem::Logger::debug(message.str());
  specfem::Logger::debug(
      "Number of material systems = " + std::to_string(numat) + "\n\n");

  // Section for acoustic isotropic
  std::vector<specfem::medium::material<specfem::dimension::type::dim2,
                                        acoustic, isotropic> >
      l_acoustic_isotropic;

  l_acoustic_isotropic.reserve(numat);

  int index_acoustic_isotropic = 0;

  // Section for elastic isotropic
  std::vector<specfem::medium::material<specfem::dimension::type::dim2,
                                        elastic_psv, isotropic> >
      l_elastic_psv_isotropic;

  std::vector<specfem::medium::material<specfem::dimension::type::dim2,
                                        elastic_sh, isotropic> >
      l_elastic_sh_isotropic;

  l_elastic_psv_isotropic.reserve(numat);
  l_elastic_sh_isotropic.reserve(numat);

  int index_elastic_isotropic = 0;

  // Section for elastic anisotropic
  std::vector<specfem::medium::material<specfem::dimension::type::dim2,
                                        elastic_psv, anisotropic> >
      l_elastic_psv_anisotropic;

  std::vector<specfem::medium::material<specfem::dimension::type::dim2,
                                        elastic_sh, anisotropic> >
      l_elastic_sh_anisotropic;

  l_elastic_psv_anisotropic.reserve(numat);
  l_elastic_sh_anisotropic.reserve(numat);

  int index_elastic_anisotropic = 0;

  // Section for elastic isotropic cosserat medium psv
  std::vector<specfem::medium::material<specfem::dimension::type::dim2,
                                        elastic_psv_t, isotropic_cosserat> >
      l_elastic_psv_t_isotropic_cosserat;

  l_elastic_psv_t_isotropic_cosserat.reserve(numat);

  int index_elastic_psv_t_isotropic_cosserat = 0;

  // Section for poroelastic isotropic
  std::vector<specfem::medium::material<specfem::dimension::type::dim2,
                                        poroelastic, isotropic> >
      l_poroelastic_isotropic;
  l_poroelastic_isotropic.reserve(numat);
  int index_poroelastic_isotropic = 0;

  // Section for electromagnetic isotropic
  std::vector<specfem::medium::material<specfem::dimension::type::dim2,
                                        electromagnetic_te, isotropic> >
      l_electromagnetic_te_isotropic;

  l_electromagnetic_te_isotropic.reserve(numat);

  int index_electromagnetic_te_isotropic = 0;

  // Loop over number of materials and read material properties
  for (int i = 0; i < numat; i++) {

    specfem::io::fortran_read_line(
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
        const type_real density = static_cast<type_real>(read_values.val0);
        const type_real cp = static_cast<type_real>(read_values.val1);
        const type_real compaction_grad =
            static_cast<type_real>(read_values.val3);
        const type_real Qkappa = static_cast<type_real>(read_values.val5);
        const type_real Qmu = static_cast<type_real>(read_values.val6);

        specfem::medium::material<specfem::dimension::type::dim2, acoustic,
                                  isotropic>
            acoustic_isotropic_holder(density, cp, Qkappa, Qmu,
                                      compaction_grad);

        acoustic_isotropic_holder.print();

        l_acoustic_isotropic.push_back(acoustic_isotropic_holder);

        index_mapping[i] = specfem::mesh::
            materials<specfem::dimension::type::dim2>::material_specification(
                specfem::element::medium_tag::acoustic,
                specfem::element::property_tag::isotropic,
                index_acoustic_isotropic, read_values.n - 1);

        index_acoustic_isotropic++;

      } else {

        const type_real density = static_cast<type_real>(read_values.val0);
        const type_real cp = static_cast<type_real>(read_values.val1);
        const type_real cs = static_cast<type_real>(read_values.val2);
        const type_real compaction_grad =
            static_cast<type_real>(read_values.val3);
        const type_real Qkappa = static_cast<type_real>(read_values.val5);
        const type_real Qmu = static_cast<type_real>(read_values.val6);

        if (elastic_wave == specfem::enums::elastic_wave::psv) {
          specfem::medium::material<specfem::dimension::type::dim2, elastic_psv,
                                    isotropic>
              elastic_isotropic_holder(density, cs, cp, Qkappa, Qmu,
                                       compaction_grad);

          elastic_isotropic_holder.print();
          l_elastic_psv_isotropic.push_back(elastic_isotropic_holder);
          index_mapping[i] = specfem::mesh::
              materials<specfem::dimension::type::dim2>::material_specification(
                  specfem::element::medium_tag::elastic_psv,
                  specfem::element::property_tag::isotropic,
                  index_elastic_isotropic, read_values.n - 1);
        } else {
          specfem::medium::material<specfem::dimension::type::dim2, elastic_sh,
                                    isotropic>
              elastic_isotropic_holder(density, cs, cp, Qkappa, Qmu,
                                       compaction_grad);

          elastic_isotropic_holder.print();
          l_elastic_sh_isotropic.push_back(elastic_isotropic_holder);
          index_mapping[i] = specfem::mesh::
              materials<specfem::dimension::type::dim2>::material_specification(
                  specfem::element::medium_tag::elastic_sh,
                  specfem::element::property_tag::isotropic,
                  index_elastic_isotropic, read_values.n - 1);
        }

        // index_mapping[i] = specfem::mesh::materials::material_specification(
        //     elastic, specfem::element::property_tag::isotropic,
        //     index_elastic_isotropic, read_values.n - 1);

        index_elastic_isotropic++;
      }
    }

    // Ansotropic material
    else if (read_values.indic == 2) {
      const type_real density = static_cast<type_real>(read_values.val0);
      const type_real c11 = static_cast<type_real>(read_values.val1);
      const type_real c13 = static_cast<type_real>(read_values.val2);
      const type_real c15 = static_cast<type_real>(read_values.val3);
      const type_real c33 = static_cast<type_real>(read_values.val4);
      const type_real c35 = static_cast<type_real>(read_values.val5);
      const type_real c55 = static_cast<type_real>(read_values.val6);
      const type_real c12 = static_cast<type_real>(read_values.val7);
      const type_real c23 = static_cast<type_real>(read_values.val8);
      const type_real c25 = static_cast<type_real>(read_values.val9);
      const type_real Qkappa = static_cast<type_real>(read_values.val11);
      const type_real Qmu = static_cast<type_real>(read_values.val12);

      if (elastic_wave == specfem::enums::elastic_wave::psv) {

        specfem::medium::material<specfem::dimension::type::dim2, elastic_psv,
                                  anisotropic>
            elastic_anisotropic_holder(density, c11, c13, c15, c33, c35, c55,
                                       c12, c23, c25, Qkappa, Qmu);

        elastic_anisotropic_holder.print();
        l_elastic_psv_anisotropic.push_back(elastic_anisotropic_holder);
        index_mapping[i] = specfem::mesh::
            materials<specfem::dimension::type::dim2>::material_specification(
                specfem::element::medium_tag::elastic_psv,
                specfem::element::property_tag::anisotropic,
                index_elastic_anisotropic, read_values.n - 1);
      } else {

        specfem::medium::material<specfem::dimension::type::dim2, elastic_sh,
                                  anisotropic>
            elastic_anisotropic_holder(density, c11, c13, c15, c33, c35, c55,
                                       c12, c23, c25, Qkappa, Qmu);

        elastic_anisotropic_holder.print();
        l_elastic_sh_anisotropic.push_back(elastic_anisotropic_holder);
        index_mapping[i] = specfem::mesh::
            materials<specfem::dimension::type::dim2>::material_specification(
                specfem::element::medium_tag::elastic_sh,
                specfem::element::property_tag::anisotropic,
                index_elastic_anisotropic, read_values.n - 1);
      }

      // index_mapping[i] = specfem::mesh::materials::material_specification(
      //     elastic, specfem::element::property_tag::anisotropic,
      //     index_elastic_anisotropic, read_values.n - 1);

      index_elastic_anisotropic++;

    } else if (read_values.indic == 3) {
      const type_real rhos = static_cast<type_real>(read_values.val0);
      const type_real rhof = static_cast<type_real>(read_values.val1);
      const type_real phi = static_cast<type_real>(read_values.val2);
      const type_real c = static_cast<type_real>(read_values.val3);
      const type_real kxx = static_cast<type_real>(read_values.val4);
      const type_real kxz = static_cast<type_real>(read_values.val5);
      const type_real kzz = static_cast<type_real>(read_values.val6);
      const type_real Ks = static_cast<type_real>(read_values.val7);
      const type_real Kf = static_cast<type_real>(read_values.val8);
      const type_real Kfr = static_cast<type_real>(read_values.val9);
      const type_real etaf = static_cast<type_real>(read_values.val10);
      const type_real mufr = static_cast<type_real>(read_values.val11);
      const type_real Qmu = static_cast<type_real>(read_values.val12);

      specfem::medium::material<specfem::dimension::type::dim2, poroelastic,
                                isotropic>
          poroelastic_isotropic_holder(rhos, rhof, phi, c, kxx, kxz, kzz, Ks,
                                       Kf, Kfr, etaf, mufr, Qmu);
      poroelastic_isotropic_holder.print();

      l_poroelastic_isotropic.push_back(poroelastic_isotropic_holder);
      index_mapping[i] =
          specfem::mesh::materials<specfem::dimension::type::dim2>::
              material_specification(specfem::element::medium_tag::poroelastic,
                                     specfem::element::property_tag::isotropic,
                                     index_poroelastic_isotropic,
                                     read_values.n - 1);
      index_poroelastic_isotropic++;
    }
    // Electromagnetic material
    else if (read_values.indic == 4) {

      const type_real mu0 = static_cast<type_real>(read_values.val0);
      const type_real e0 = static_cast<type_real>(read_values.val1);
      const type_real e11 = static_cast<type_real>(read_values.val2); // e11(e0)
      const type_real e33 = static_cast<type_real>(read_values.val3); // e33(e0)
      const type_real sig11 = static_cast<type_real>(read_values.val4);
      const type_real sig33 = static_cast<type_real>(read_values.val5);
      const type_real Qe11 = static_cast<type_real>(read_values.val6);
      const type_real Qe33 = static_cast<type_real>(read_values.val7);
      const type_real Qs11 = static_cast<type_real>(read_values.val8);
      const type_real Qs33 = static_cast<type_real>(read_values.val9);

      if (elastic_wave == specfem::enums::elastic_wave::psv) {
        specfem::medium::material<specfem::dimension::type::dim2,
                                  electromagnetic_te, isotropic>
            electromagnetic_te_isotropic_holder(mu0, e0, e11, e33, sig11, sig33,
                                                Qe11, Qe33, Qs11, Qs33);

        electromagnetic_te_isotropic_holder.print();

        l_electromagnetic_te_isotropic.push_back(
            electromagnetic_te_isotropic_holder);

        index_mapping[i] = specfem::mesh::
            materials<specfem::dimension::type::dim2>::material_specification(
                specfem::element::medium_tag::electromagnetic_te,
                specfem::element::property_tag::isotropic,
                index_electromagnetic_te_isotropic, read_values.n - 1);

        index_electromagnetic_te_isotropic++;
      }
    } else if (read_values.indic == 5) {

      if (elastic_wave == specfem::enums::elastic_wave::psv) {
        const type_real rho = static_cast<type_real>(read_values.val0);
        const type_real kappa = static_cast<type_real>(read_values.val1);
        const type_real mu = static_cast<type_real>(read_values.val2);
        const type_real nu = static_cast<type_real>(read_values.val3);
        const type_real j = static_cast<type_real>(read_values.val4);
        const type_real lambda_c = static_cast<type_real>(read_values.val5);
        const type_real mu_c = static_cast<type_real>(read_values.val6);
        const type_real nu_c = static_cast<type_real>(read_values.val7);

        // Create the material
        specfem::medium::material<specfem::dimension::type::dim2, elastic_psv_t,
                                  isotropic_cosserat>
            elastic_psv_t_isotropic_cosserat_holder(rho, kappa, mu, nu, j,
                                                    lambda_c, mu_c, nu_c);
        // Print the material properties
        elastic_psv_t_isotropic_cosserat_holder.print();

        // Add the material to the list
        l_elastic_psv_t_isotropic_cosserat.push_back(
            elastic_psv_t_isotropic_cosserat_holder);

        // Add the material to the index mapping
        index_mapping[i] = specfem::mesh::
            materials<specfem::dimension::type::dim2>::material_specification(
                specfem::element::medium_tag::elastic_psv_t,
                specfem::element::property_tag::isotropic_cosserat,
                index_elastic_psv_t_isotropic_cosserat, read_values.n - 1);
        index_elastic_psv_t_isotropic_cosserat++;

      } else {
        std::ostringstream message;
        message << "Elastic Isotropic Cosserat medium not supported for SH ["
                << __FILE__ << ":" << __LINE__ << "]\n";
        throw std::runtime_error(message.str());
      }

    } else {
      std::ostringstream message;
      message << "Material type " << read_values.indic << " not supported ["
              << __FILE__ << ":" << __LINE__ << "]\n";
      throw std::runtime_error(message.str());
    }
  }
  // Sum materials and check if the total number of materials is correct
  int total_materials =
      l_acoustic_isotropic.size() + l_elastic_psv_isotropic.size() +
      l_elastic_sh_isotropic.size() + l_elastic_psv_anisotropic.size() +
      l_elastic_sh_anisotropic.size() + l_poroelastic_isotropic.size() +
      l_electromagnetic_te_isotropic.size() +
      l_elastic_psv_t_isotropic_cosserat.size();
  if (total_materials != numat) {
    std::ostringstream message;
    message << "Total number of materials not matching the input materials ["
            << __FILE__ << ":" << __LINE__ << "]\n"
            << "Total number of materials: " << numat << "\n"
            << "  acoustic isotropic:............ "
            << l_acoustic_isotropic.size() << "\n"
            << "  elastic isotropic psv:.......... "
            << l_elastic_psv_isotropic.size() << "\n"
            << "  elastic isotropic sh:.......... "
            << l_elastic_sh_isotropic.size() << "\n"
            << "  elastic anisotropic psv:........ "
            << l_elastic_psv_anisotropic.size() << "\n"
            << "  elastic anisotropic sh:........ "
            << l_elastic_sh_anisotropic.size() << "\n"
            << "  poroelastic isotropic:......... "
            << l_poroelastic_isotropic.size() << "\n"
            << "  electromagnetic_te isotropic:.. "
            << l_electromagnetic_te_isotropic.size() << "\n";
    throw std::runtime_error(message.str());
  }

  // Create materials instances
  acoustic_isotropic =
      specfem::mesh::materials<specfem::dimension::type::dim2>::material<
          acoustic, isotropic>(l_acoustic_isotropic.size(),
                               l_acoustic_isotropic);

  elastic_psv_isotropic =
      specfem::mesh::materials<specfem::dimension::type::dim2>::material<
          elastic_psv, isotropic>(l_elastic_psv_isotropic.size(),
                                  l_elastic_psv_isotropic);

  elastic_sh_isotropic =
      specfem::mesh::materials<specfem::dimension::type::dim2>::material<
          elastic_sh, isotropic>(l_elastic_sh_isotropic.size(),
                                 l_elastic_sh_isotropic);

  elastic_psv_anisotropic =
      specfem::mesh::materials<specfem::dimension::type::dim2>::material<
          elastic_psv, anisotropic>(l_elastic_psv_anisotropic.size(),
                                    l_elastic_psv_anisotropic);

  elastic_sh_anisotropic =
      specfem::mesh::materials<specfem::dimension::type::dim2>::material<
          elastic_sh, anisotropic>(l_elastic_sh_anisotropic.size(),
                                   l_elastic_sh_anisotropic);

  poroelastic_isotropic =
      specfem::mesh::materials<specfem::dimension::type::dim2>::material<
          poroelastic, isotropic>(l_poroelastic_isotropic.size(),
                                  l_poroelastic_isotropic);

  electromagnetic_te_isotropic =
      specfem::mesh::materials<specfem::dimension::type::dim2>::material<
          electromagnetic_te, isotropic>(l_electromagnetic_te_isotropic.size(),
                                         l_electromagnetic_te_isotropic);

  elastic_psv_t_isotropic_cosserat =
      specfem::mesh::materials<specfem::dimension::type::dim2>::material<
          elastic_psv_t, isotropic_cosserat>(
          l_elastic_psv_t_isotropic_cosserat.size(),
          l_elastic_psv_t_isotropic_cosserat);
  return index_mapping;
}

void read_material_indices(
    std::ifstream &stream, const int nspec, const int numat,
    const std::vector<specfem::mesh::materials<
        specfem::dimension::type::dim2>::material_specification> &index_mapping,
    const specfem::kokkos::HostView1d<specfem::mesh::materials<
        specfem::dimension::type::dim2>::material_specification>
        material_index_mapping,
    const specfem::kokkos::HostView2d<int> knods,
    const specfem::MPI::MPI *mpi) {

  const int ngnod = knods.extent(0);

  int n, kmato_read, pml_read;

  std::vector<int> knods_read(ngnod, -1);

  for (int ispec = 0; ispec < nspec; ispec++) {
    // format: #element_id  #material_id #node_id1 #node_id2 #...
    specfem::io::fortran_read_line(stream, &n, &kmato_read, &knods_read,
                                   &pml_read);

    if (n < 1 || n > nspec) {
      std::ostringstream message;
      message << "Error reading material indices: element index out of bounds, "
                 "read "
              << n << " but should be between 1 and " << nspec;
      throw std::runtime_error(message.str());
    }

    if (kmato_read < 1 || kmato_read > numat) {
      std::ostringstream message;
      message
          << "Error reading material indices: material index out of bounds, "
             "read "
          << kmato_read << " but should be between 1 and " << numat;
      throw std::runtime_error(message.str());
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

specfem::mesh::materials<specfem::dimension::type::dim2>
specfem::io::mesh::impl::fortran::dim2::read_material_properties(
    std::ifstream &stream, const int numat, const int nspec,
    const specfem::enums::elastic_wave elastic_wave,
    const specfem::enums::electromagnetic_wave electromagnetic_wave,
    const specfem::kokkos::HostView2d<int> knods,
    const specfem::MPI::MPI *mpi) {

  // Create materials instances
  specfem::mesh::materials<specfem::dimension::type::dim2> materials(nspec,
                                                                     numat);

  // Read material properties
  auto index_mapping = read_materials(
      stream, numat, elastic_wave, electromagnetic_wave,
      materials.get_container<acoustic, isotropic>(),
      materials.get_container<elastic_psv, isotropic>(),
      materials.get_container<elastic_sh, isotropic>(),
      materials.get_container<elastic_psv, anisotropic>(),
      materials.get_container<elastic_sh, anisotropic>(),
      materials.get_container<poroelastic, isotropic>(),
      materials.get_container<electromagnetic_te, isotropic>(),
      materials.get_container<elastic_psv_t, isotropic_cosserat>(), mpi);

  // Read material indices
  read_material_indices(stream, nspec, numat, index_mapping,
                        materials.material_index_mapping, knods, mpi);

  return materials;
}
