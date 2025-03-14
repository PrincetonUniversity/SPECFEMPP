
#include "../test_fixture/test_fixture.hpp"
#include "mesh/mesh.hpp"
#include <cmath>
#include <unordered_map>
#include <variant>
#include <vector>

using MaterialVectorType = std::vector<std::variant<
    specfem::medium::material<specfem::element::medium_tag::acoustic,
                              specfem::element::property_tag::isotropic>,
    specfem::medium::material<specfem::element::medium_tag::elastic_sv,
                              specfem::element::property_tag::isotropic>,
    specfem::medium::material<specfem::element::medium_tag::elastic_sh,
                              specfem::element::property_tag::isotropic>,
    specfem::medium::material<specfem::element::medium_tag::elastic_sv,
                              specfem::element::property_tag::anisotropic>,
    specfem::medium::material<specfem::element::medium_tag::elastic_sh,
                              specfem::element::property_tag::anisotropic>,
    specfem::medium::material<specfem::element::medium_tag::electromagnetic_sv,
                              specfem::element::property_tag::isotropic> > >;

const static std::unordered_map<std::string, MaterialVectorType>
    ground_truth = {
      { "Simple mesh with flat topography (P_SV wave)",
        MaterialVectorType({ specfem::medium::material<
            specfem::element::medium_tag::elastic_sv,
            specfem::element::property_tag::isotropic>(2700.0, 1732.051, 3000.0,
                                                       9999, 9999, 0.0) }) },
      { "Simple mesh with flat topography (SH wave)",
        MaterialVectorType({ specfem::medium::material<
            specfem::element::medium_tag::elastic_sh,
            specfem::element::property_tag::isotropic>(2700.0, 1732.051, 3000.0,
                                                       9999, 9999, 0.0) }) },
      { "Simple mesh with curved topography",
        MaterialVectorType({ specfem::medium::material<
            specfem::element::medium_tag::elastic_sv,
            specfem::element::property_tag::isotropic>(2700.0, 1732.051, 3000.0,
                                                       9999, 9999, 0.0) }) },
      { "Simple mesh with flat ocean bottom",
        MaterialVectorType({ specfem::medium::material<
                                 specfem::element::medium_tag::elastic_sv,
                                 specfem::element::property_tag::isotropic>(
                                 2500.0, 1963.0, 3400.0, 9999, 9999, 0.0),
                             specfem::medium::material<
                                 specfem::element::medium_tag::acoustic,
                                 specfem::element::property_tag::isotropic>(
                                 1020.0, 1500, 9999, 9999, 0.0)

        }) },
      { "Simple mesh with curved ocean bottom",
        MaterialVectorType({ specfem::medium::material<
                                 specfem::element::medium_tag::elastic_sv,
                                 specfem::element::property_tag::isotropic>(
                                 2500.0, 1963.0, 3400.0, 9999, 9999, 0.0),
                             specfem::medium::material<
                                 specfem::element::medium_tag::acoustic,
                                 specfem::element::property_tag::isotropic>(
                                 1020.0, 1500, 9999, 9999, 0.0)

        }) },
      { "Gmesh Example",
        MaterialVectorType({ specfem::medium::material<
                                 specfem::element::medium_tag::acoustic,
                                 specfem::element::property_tag::isotropic>(
                                 2000.0, 1680.0, 10.0, 10.0, 0.0),
                             specfem::medium::material<
                                 specfem::element::medium_tag::acoustic,
                                 specfem::element::property_tag::isotropic>(
                                 1000.0, 1477.0, 10.0, 10.0, 0.0) }) },

      { "Homogeneous Elastic Anisotropic Material (P_SV wave)",
        MaterialVectorType({ specfem::medium::material<
            specfem::element::medium_tag::elastic_sv,
            specfem::element::property_tag::anisotropic>(
            2700.0, 24299994600.5, 8099996400.35, 0.0, 24299994600.5, 0.0,
            8100001799.8227, 8099996400.35, 8099996400.35, 0.0, 9999,
            9999) }) },
      { "Homogeneous Elastic Anisotropic Material (SH wave)",
        MaterialVectorType({ specfem::medium::material<
            specfem::element::medium_tag::elastic_sh,
            specfem::element::property_tag::anisotropic>(
            2700.0, 24299994600.5, 8099996400.35, 0.0, 24299994600.5, 0.0,
            8100001799.8227, 8099996400.35, 8099996400.35, 0.0, 9999,
            9999) }) },
      { "Electro-magnetic mesh example from Morency 2020",
        MaterialVectorType(
            { specfem::medium::material<
                  specfem::element::medium_tag::electromagnetic_sv,
                  specfem::element::property_tag::isotropic>(
                  12.566 * std::pow(10, -7), 8.85 * std::pow(10, -12), 5.0, 5.0,
                  2.0 * std::pow(10, -3), 2.0 * std::pow(10, -3), 90.0, 90.0,
                  90.0, 90.0),
              specfem::medium::material<
                  specfem::element::medium_tag::electromagnetic_sv,
                  specfem::element::property_tag::isotropic>(
                  12.566 * std::pow(10, -7), 8.85 * std::pow(10, -12), 1.0, 1.0,
                  0.0 * std::pow(10, -3), 0.0 * std::pow(10, -3), 90.0, 90.0,
                  90.0, 90.0) }) }
    };

void check_test(
    const specfem::mesh::materials<specfem::dimension::type::dim2> &computed,
    const MaterialVectorType &expected) {

  if (computed.n_materials != expected.size()) {
    throw std::runtime_error("Size of materials is not the same");
  }

  const int n_materials = computed.n_materials;

  if (n_materials == 0) {
    return;
  }

  const int nspec = computed.material_index_mapping.extent(0);

  for (int ispec = 0; ispec < nspec; ispec++) {
    const auto material_specification = computed.material_index_mapping(ispec);

    const auto type = material_specification.type;
    const auto property = material_specification.property;
    const int index = material_specification.index;
    const int imaterial = material_specification.database_index;

    // Acoustic Isotropic
    if ((type == specfem::element::medium_tag::acoustic) &&
        (property == specfem::element::property_tag::isotropic)) {
      const auto icomputed = std::get<specfem::medium::material<
          specfem::element::medium_tag::acoustic,
          specfem::element::property_tag::isotropic> >(computed[ispec]);
      const auto iexpected = std::get<specfem::medium::material<
          specfem::element::medium_tag::acoustic,
          specfem::element::property_tag::isotropic> >(expected[imaterial]);
      if (icomputed != iexpected) {
        std::ostringstream error_message;
        error_message << "Material " << index << " is not the same ["
                      << __FILE__ << ":" << __LINE__ << "]\n"
                      << "  imaterial: " << imaterial << "\n"
                      << "  index:     " << index << "\n"
                      << "  ispec:     " << ispec << "\n"
                      << "Computed: \n"
                      << icomputed.print() << "\n"
                      << "Expected: \n"
                      << iexpected.print() << "\n";
        throw std::runtime_error(error_message.str());
      }
    }
    // Elastic SV Isoptropic
    else if ((type == specfem::element::medium_tag::elastic_sv) &&
             (property == specfem::element::property_tag::isotropic)) {
      const auto icomputed = std::get<specfem::medium::material<
          specfem::element::medium_tag::elastic_sv,
          specfem::element::property_tag::isotropic> >(computed[ispec]);
      const auto iexpected = std::get<specfem::medium::material<
          specfem::element::medium_tag::elastic_sv,
          specfem::element::property_tag::isotropic> >(expected[imaterial]);
      if (icomputed != iexpected) {
        std::ostringstream error_message;
        error_message << "Material " << index << " is not the same ["
                      << __FILE__ << ":" << __LINE__ << "]\n"
                      << "  imaterial: " << imaterial << "\n"
                      << "  index:     " << index << "\n"
                      << "  ispec:     " << ispec << "\n"
                      << "Computed: \n"
                      << icomputed.print() << "\n"
                      << "Expected: \n"
                      << iexpected.print() << "\n";
        throw std::runtime_error(error_message.str());
      }
    }
    // Elastic SH Isoptropic
    else if ((type == specfem::element::medium_tag::elastic_sh) &&
             (property == specfem::element::property_tag::isotropic)) {
      const auto icomputed = std::get<specfem::medium::material<
          specfem::element::medium_tag::elastic_sh,
          specfem::element::property_tag::isotropic> >(computed[ispec]);
      const auto iexpected = std::get<specfem::medium::material<
          specfem::element::medium_tag::elastic_sh,
          specfem::element::property_tag::isotropic> >(expected[imaterial]);
      if (icomputed != iexpected) {
        std::ostringstream error_message;
        error_message << "Material " << index << " is not the same ["
                      << __FILE__ << ":" << __LINE__ << "]\n"
                      << "  imaterial: " << imaterial << "\n"
                      << "  index:     " << index << "\n"
                      << "  ispec:     " << ispec << "\n"
                      << "Computed: \n"
                      << icomputed.print() << "\n"
                      << "Expected: \n"
                      << iexpected.print() << "\n";
        throw std::runtime_error(error_message.str());
      }
    }
    // Elastic SV Anisotropic
    else if ((type == specfem::element::medium_tag::elastic_sv) &&
             (property == specfem::element::property_tag::anisotropic)) {
      const auto icomputed = std::get<specfem::medium::material<
          specfem::element::medium_tag::elastic_sv,
          specfem::element::property_tag::anisotropic> >(computed[ispec]);
      const auto iexpected = std::get<specfem::medium::material<
          specfem::element::medium_tag::elastic_sv,
          specfem::element::property_tag::anisotropic> >(expected[imaterial]);
      if (icomputed != iexpected) {
        std::ostringstream error_message;
        error_message << "Material " << index << " is not the same ["
                      << __FILE__ << ":" << __LINE__ << "]\n"
                      << "  imaterial: " << imaterial << "\n"
                      << "  index:     " << index << "\n"
                      << "  ispec:     " << ispec << "\n"
                      << "Computed: \n"
                      << icomputed.print() << "\n"
                      << "Expected: \n"
                      << iexpected.print() << "\n";
        throw std::runtime_error(error_message.str());
      }
      // Elastic SH Anisotropic
    } else if ((type == specfem::element::medium_tag::elastic_sh) &&
               (property == specfem::element::property_tag::anisotropic)) {
      const auto icomputed = std::get<specfem::medium::material<
          specfem::element::medium_tag::elastic_sh,
          specfem::element::property_tag::anisotropic> >(computed[ispec]);
      const auto iexpected = std::get<specfem::medium::material<
          specfem::element::medium_tag::elastic_sh,
          specfem::element::property_tag::anisotropic> >(expected[imaterial]);
      if (icomputed != iexpected) {
        std::ostringstream error_message;
        error_message << "Material " << index << " is not the same ["
                      << __FILE__ << ":" << __LINE__ << "]\n"
                      << "  imaterial: " << imaterial << "\n"
                      << "  index:     " << index << "\n"
                      << "  ispec:     " << ispec << "\n"
                      << "Computed: \n"
                      << icomputed.print() << "\n"
                      << "Expected: \n"
                      << iexpected.print() << "\n";
        throw std::runtime_error(error_message.str());
      }
    }
    // Electromagnetic SV Isotropic
    else if ((type == specfem::element::medium_tag::electromagnetic_sv) &&
             (property == specfem::element::property_tag::isotropic)) {
      const auto icomputed = std::get<specfem::medium::material<
          specfem::element::medium_tag::electromagnetic_sv,
          specfem::element::property_tag::isotropic> >(computed[ispec]);
      const auto iexpected = std::get<specfem::medium::material<
          specfem::element::medium_tag::electromagnetic_sv,
          specfem::element::property_tag::isotropic> >(expected[imaterial]);
      if (icomputed != iexpected) {
        std::ostringstream error_message;
        error_message << "Material " << index << " is not the same ["
                      << __FILE__ << ":" << __LINE__ << "]\n"
                      << "  ispec:     " << ispec << "\n"
                      << "  imaterial: " << imaterial << "\n"
                      << "  index:     " << index << "\n"
                      << "Computed: \n"
                      << icomputed.print() << "\n"
                      << "Expected: \n"
                      << iexpected.print() << "\n";
        throw std::runtime_error(error_message.str());
      }
    } else {
      throw std::runtime_error("Material type not supported");
    }
  }

  return;
}

TEST_F(MESH, materials) {
  for (auto parameters : *this) {
    const auto Test = std::get<0>(parameters);
    auto mesh = std::get<1>(parameters);

    try {

      const auto computed = mesh.materials;
      const auto expected = ground_truth.at(Test.name);

      check_test(computed, expected);

      std::cout << "-------------------------------------------------------\n"
                << "\033[0;32m[PASSED]\033[0m Test " << Test.number << ": "
                << Test.name << "\n"
                << "-------------------------------------------------------\n\n"
                << std::endl;
    } catch (std::exception &e) {
      std::cout << "-------------------------------------------------------\n"
                << "\033[0;31m[FAILED]\033[0m \n"
                << "-------------------------------------------------------\n"
                << "- Test " << Test.number << ": " << Test.name << "\n"
                << "- Error: " << e.what() << "\n"
                << "-------------------------------------------------------\n\n"
                << std::endl;
      ADD_FAILURE();
    }
  }
}
