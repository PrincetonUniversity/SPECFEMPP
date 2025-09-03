
#include "../test_fixture/test_fixture.hpp"
#include "mesh/mesh.hpp"
#include <cmath>
#include <unordered_map>
#include <variant>
#include <vector>

using MaterialVectorType = std::vector<std::any>; /// NOLINT

const static std::unordered_map<std::string, MaterialVectorType>
    material_ground_truth = {
      { "Simple mesh with flat topography (P_SV wave)",
        MaterialVectorType({ specfem::medium::material<
            specfem::element::medium_tag::elastic_psv,
            specfem::element::property_tag::isotropic>(2700.0, 1732.051, 3000.0,
                                                       9999, 9999, 0.0) }) },
      { "Simple mesh with flat topography (SH wave)",
        MaterialVectorType({ specfem::medium::material<
            specfem::element::medium_tag::elastic_sh,
            specfem::element::property_tag::isotropic>(2700.0, 1732.051, 3000.0,
                                                       9999, 9999, 0.0) }) },
      { "Simple mesh with curved topography",
        MaterialVectorType({ specfem::medium::material<
            specfem::element::medium_tag::elastic_psv,
            specfem::element::property_tag::isotropic>(2700.0, 1732.051, 3000.0,
                                                       9999, 9999, 0.0) }) },
      { "Simple mesh with flat ocean bottom",
        MaterialVectorType({ specfem::medium::material<
                                 specfem::element::medium_tag::elastic_psv,
                                 specfem::element::property_tag::isotropic>(
                                 2500.0, 1963.0, 3400.0, 9999, 9999, 0.0),
                             specfem::medium::material<
                                 specfem::element::medium_tag::acoustic,
                                 specfem::element::property_tag::isotropic>(
                                 1020.0, 1500, 9999, 9999, 0.0)

        }) },
      { "Simple mesh with curved ocean bottom",
        MaterialVectorType({ specfem::medium::material<
                                 specfem::element::medium_tag::elastic_psv,
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
            specfem::element::medium_tag::elastic_psv,
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
      { "Poroelastic mesh - Homogeneous isotropic material",
        MaterialVectorType({ specfem::medium::material<
            specfem::element::medium_tag::poroelastic,
            specfem::element::property_tag::isotropic>(
            2650.0, 880.0, 0.1, 2.0, 1.0e-9, 0.0, 1.0e-9, 12.2e9, 1.985e9,
            9.6e9, 0.0, 5.1e9, 9999) }) },
      { "Electro-magnetic mesh example from Morency 2020",
        MaterialVectorType(
            { specfem::medium::material<
                  specfem::element::medium_tag::electromagnetic_te,
                  specfem::element::property_tag::isotropic>(
                  12.566 * std::pow(10, -7), 8.85 * std::pow(10, -12), 5.0, 5.0,
                  2.0 * std::pow(10, -3), 2.0 * std::pow(10, -3), 90.0, 90.0,
                  90.0, 90.0),
              specfem::medium::material<
                  specfem::element::medium_tag::electromagnetic_te,
                  specfem::element::property_tag::isotropic>(
                  12.566 * std::pow(10, -7), 8.85 * std::pow(10, -12), 1.0, 1.0,
                  0.0 * std::pow(10, -3), 0.0 * std::pow(10, -3), 90.0, 90.0,
                  90.0, 90.0) }) },
      { "Elastic Isotropic Cosserat Medium - Homogeneous",
        MaterialVectorType({ specfem::medium::material<
            specfem::element::medium_tag::elastic_psv_t,
            specfem::element::property_tag::isotropic_cosserat>(
            2700.0, 13.5e9, 8.1e9, 2.7e5, 2700.0, 7.75e11, 1.5e11, 2.7e5) }) }
    };

void check_material(
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

    const auto medium_tag = material_specification.type;
    const auto property_tag = material_specification.property;
    const int index = material_specification.index;
    const int imaterial = material_specification.database_index;

    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM2),
         MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC,
                    ELASTIC_PSV_T, ELECTROMAGNETIC_TE),
         PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT)),
        {
          if ((medium_tag == _medium_tag_) &&
              (property_tag == _property_tag_)) {
            const auto icomputed =
                computed.get_material<_medium_tag_, _property_tag_>(ispec);
            const auto iexpected = std::any_cast<
                specfem::medium::material<_medium_tag_, _property_tag_> >(
                expected[imaterial]);
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
            return;
          }
        })

    // If we reach here, the material type is not supported
    std::ostringstream error_message;
    error_message << "Material type not supported [" << __FILE__ << ":"
                  << __LINE__ << "]\n"
                  << "  imaterial: " << imaterial << "\n"
                  << "  index:     " << index << "\n"
                  << "  ispec:     " << ispec << "\n";
    throw std::runtime_error(error_message.str());
  }

  return;
}

TEST_F(MESH, materials) {
  for (auto parameters : *this) {
    const auto Test = std::get<0>(parameters);
    auto mesh = std::get<1>(parameters);

    try {

      const auto computed = mesh.materials;
      const auto expected = material_ground_truth.at(Test.name);

      check_material(computed, expected);

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
