#include "point/properties.hpp"
#include "../test_fixture/test_fixture.hpp"
#include "mesh/mesh.hpp"
#include <cmath>
#include <unordered_map>
#include <variant>
#include <vector>

#define MEDIUM_TYPE(POSTFIX, DIMENSION_TAG, MEDIUM_TAG, PROPERTY_TAG)          \
  using type_##POSTFIX = specfem::point::properties<DIMENSION_TAG, MEDIUM_TAG, \
                                                    PROPERTY_TAG, false>;

CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(
    WHERE(DIMENSION_TAG_DIM2)
        WHERE(MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ELASTIC_SH, MEDIUM_TAG_ACOUSTIC,
              MEDIUM_TAG_POROELASTIC)
            WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC),
    MEDIUM_TYPE)

#undef MEDIUM_TYPE

#define TYPE_NAME(POSTFIX, ...) (type_##POSTFIX)

#define MAKE_VARIANT_RETURN                                                    \
  std::variant<BOOST_PP_SEQ_ENUM(CALL_MACRO_FOR_ALL_MATERIAL_SYSTEMS(          \
      WHERE(DIMENSION_TAG_DIM2)                                                \
          WHERE(MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ELASTIC_SH,                  \
                MEDIUM_TAG_ACOUSTIC, MEDIUM_TAG_POROELASTIC)                   \
              WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC),         \
      TYPE_NAME))>

using MaterialVectorType = std::vector<MAKE_VARIANT_RETURN>; /// NOLINT

#undef MAKE_VARIANT_RETURN
#undef TYPE_NAME

constexpr static auto dimension = specfem::dimension::type::dim2;

const static std::unordered_map<std::string, MaterialVectorType>
    ground_truth = {
      { "Simple mesh with flat topography (P_SV wave)",
        MaterialVectorType({ specfem::point::properties<
            dimension, specfem::element::medium_tag::elastic_sv,
            specfem::element::property_tag::isotropic, false>(
            static_cast<type_real>(24300000000.0),
            static_cast<type_real>(8100001799.82),
            static_cast<type_real>(2700.0)) }) },
      { "Simple mesh with flat topography (SH wave)",
        MaterialVectorType({ specfem::point::properties<
            dimension, specfem::element::medium_tag::elastic_sh,
            specfem::element::property_tag::isotropic, false>(
            static_cast<type_real>(24300000000.0),
            static_cast<type_real>(8100001799.82),
            static_cast<type_real>(2700.0)) }) },
      { "Simple mesh with curved topography",
        MaterialVectorType({ specfem::point::properties<
            dimension, specfem::element::medium_tag::elastic_sv,
            specfem::element::property_tag::isotropic, false>(
            static_cast<type_real>(24300000000.0),
            static_cast<type_real>(8100001799.82),
            static_cast<type_real>(2700.0)) }) },
      { "Simple mesh with flat ocean bottom",
        MaterialVectorType(
            { specfem::point::properties<
                  dimension, specfem::element::medium_tag::elastic_sv,
                  specfem::element::property_tag::isotropic, false>(
                  static_cast<type_real>(28900000000.0),
                  static_cast<type_real>(9633422500.0),
                  static_cast<type_real>(2500.0)),
              specfem::point::properties<
                  dimension, specfem::element::medium_tag::acoustic,
                  specfem::element::property_tag::isotropic, false>(
                  static_cast<type_real>(0.00098039215),
                  static_cast<type_real>(2295000000.0))

            }) },
      { "Simple mesh with curved ocean bottom",
        MaterialVectorType(
            { specfem::point::properties<
                  dimension, specfem::element::medium_tag::elastic_sv,
                  specfem::element::property_tag::isotropic, false>(
                  static_cast<type_real>(28900000000.0),
                  static_cast<type_real>(9633422500.0),
                  static_cast<type_real>(2500.0)),
              specfem::point::properties<
                  dimension, specfem::element::medium_tag::acoustic,
                  specfem::element::property_tag::isotropic, false>(
                  static_cast<type_real>(0.00098039215),
                  static_cast<type_real>(2295000000.0))

            }) },
      { "Gmesh Example",
        MaterialVectorType(
            { specfem::point::properties<
                  dimension, specfem::element::medium_tag::acoustic,
                  specfem::element::property_tag::isotropic, false>(
                  static_cast<type_real>(0.0005),
                  static_cast<type_real>(5644800000.0)),
              specfem::point::properties<
                  dimension, specfem::element::medium_tag::acoustic,
                  specfem::element::property_tag::isotropic, false>(
                  static_cast<type_real>(0.001),
                  static_cast<type_real>(2181529000.0)) }) },

      { "Homogeneous Elastic Anisotropic Material (P_SV wave)",
        MaterialVectorType({ specfem::point::properties<
            dimension, specfem::element::medium_tag::elastic_sv,
            specfem::element::property_tag::anisotropic, false>(
            static_cast<type_real>(24299994600.5),
            static_cast<type_real>(8099996400.35), static_cast<type_real>(0.0),
            static_cast<type_real>(24299994600.5), static_cast<type_real>(0.0),
            static_cast<type_real>(8100001799.8227),
            static_cast<type_real>(8099996400.35),
            static_cast<type_real>(8099996400.35), static_cast<type_real>(0.0),
            static_cast<type_real>(2700.0)) }) },
      { "Homogeneous Elastic Anisotropic Material (SH wave)",
        MaterialVectorType({ specfem::point::properties<
            dimension, specfem::element::medium_tag::elastic_sh,
            specfem::element::property_tag::anisotropic, false>(
            static_cast<type_real>(24299994600.5),
            static_cast<type_real>(8099996400.35), static_cast<type_real>(0.0),
            static_cast<type_real>(24299994600.5), static_cast<type_real>(0.0),
            static_cast<type_real>(8100001799.8227),
            static_cast<type_real>(8099996400.35),
            static_cast<type_real>(8099996400.35), static_cast<type_real>(0.0),
            static_cast<type_real>(2700.0)) }) },
      { "Poroelastic mesh - Homogeneous isotropic material",
        MaterialVectorType({ specfem::point::properties<
            dimension, specfem::element::medium_tag::poroelastic,
            specfem::element::property_tag::isotropic, false>(
            static_cast<type_real>(0.1), static_cast<type_real>(2650.0),
            static_cast<type_real>(880.0), static_cast<type_real>(2.0),
            static_cast<type_real>(5.1e9),
            static_cast<type_real>(17161412448.29),
            static_cast<type_real>(3572781488.14),
            static_cast<type_real>(16764590059.75),
            static_cast<type_real>(1.0e-9), static_cast<type_real>(0.0),
            static_cast<type_real>(1.0e-9), static_cast<type_real>(0.0)) }) },
      // { "Electro-magnetic mesh example from Morency 2020",
      //   MaterialVectorType(
      //       { specfem::point::properties<
      //             dimension,
      //             specfem::element::medium_tag::electromagnetic_sv,
      //             specfem::element::property_tag::isotropic, false>(
      //             12.566 * std::pow(10, -7), 8.85 * std::pow(10,
      //             -12), 5.0, 5.0, 2.0 * std::pow(10, -3), 2.0 * std::pow(10,
      //             -3), 90.0, 90.0, 90.0, 90.0),
      //         specfem::point::properties<
      //             dimension,
      //             specfem::element::medium_tag::electromagnetic_sv,
      //             specfem::element::property_tag::isotropic, false>(
      //             12.566 * std::pow(10, -7), 8.85 * std::pow(10,
      //             -12), 1.0, 1.0, 0.0 * std::pow(10, -3), 0.0 * std::pow(10,
      //             -3), 90.0, 90.0, 90.0, 90.0) }) }
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

    CALL_CODE_FOR_ALL_MATERIAL_SYSTEMS(
        WHERE(DIMENSION_TAG_DIM2)
            WHERE(MEDIUM_TAG_ELASTIC_SV, MEDIUM_TAG_ELASTIC_SH,
                  MEDIUM_TAG_ACOUSTIC, MEDIUM_TAG_POROELASTIC)
                WHERE(PROPERTY_TAG_ISOTROPIC, PROPERTY_TAG_ANISOTROPIC),
        if ((type == _medium_tag_) && (property == _property_tag_)) {
          const auto icomputed =
              std::get<
                  specfem::medium::material<_medium_tag_, _property_tag_> >(
                  computed[ispec])
                  .get_properties();
          const auto iexpected =
              std::get<specfem::point::properties<dimension, _medium_tag_,
                                                  _property_tag_, false> >(
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

TEST_F(MESH, derived_properties) {
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
