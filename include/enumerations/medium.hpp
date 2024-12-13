#pragma once

#include "enumerations/dimension.hpp"
#include <array>
#include <tuple>

namespace specfem {
namespace element {

constexpr int ntypes = 2; ///< Number of element types

/**
 * @brief Medium tag enumeration
 *
 */
enum class medium_tag { elastic, acoustic, poroelastic };

/**
 * @brief Property tag enumeration
 *
 */
enum class property_tag { isotropic, anisotropic };

/**
 * @brief Boundary tag enumeration
 *
 */
enum class boundary_tag {
  // primary boundaries
  none,
  acoustic_free_surface,
  stacey,

  // composite boundaries
  composite_stacey_dirichlet
};

constexpr auto medium_types() {
  constexpr int total_medium_types = 2;
  constexpr std::array<medium_tag, total_medium_types> medium_types{
    medium_tag::elastic, medium_tag::acoustic
  };
  return medium_types;
}

constexpr auto material_systems() {
  constexpr int total_material_systems = 3;
  constexpr
      std::array<std::tuple<medium_tag, property_tag>, total_material_systems>
          material_systems{
            std::make_tuple(medium_tag::elastic, property_tag::isotropic),
            std::make_tuple(medium_tag::elastic, property_tag::anisotropic),
            std::make_tuple(medium_tag::acoustic, property_tag::isotropic)
          };
  return material_systems;
}

constexpr auto element_types() {
  constexpr int total_element_types = 8;
  constexpr std::array<std::tuple<medium_tag, property_tag, boundary_tag>,
                       total_element_types>
      element_types{
        std::make_tuple(medium_tag::elastic, property_tag::isotropic,
                        boundary_tag::none),
        std::make_tuple(medium_tag::elastic, property_tag::isotropic,
                        boundary_tag::stacey),
        std::make_tuple(medium_tag::elastic, property_tag::anisotropic,
                        boundary_tag::none),
        std::make_tuple(medium_tag::elastic, property_tag::anisotropic,
                        boundary_tag::stacey),
        std::make_tuple(medium_tag::acoustic, property_tag::isotropic,
                        boundary_tag::none),
        std::make_tuple(medium_tag::acoustic, property_tag::isotropic,
                        boundary_tag::acoustic_free_surface),
        std::make_tuple(medium_tag::acoustic, property_tag::isotropic,
                        boundary_tag::stacey),
        std::make_tuple(medium_tag::acoustic, property_tag::isotropic,
                        boundary_tag::composite_stacey_dirichlet)
      };

  return element_types;
}

template <specfem::dimension::type Dimension,
          specfem::element::medium_tag MediumTag>
class attributes;

template <>
class attributes<specfem::dimension::type::dim2,
                 specfem::element::medium_tag::elastic> {

public:
  constexpr static int dimension() { return 2; }

  constexpr static int components() { return 2; }
};

template <>
class attributes<specfem::dimension::type::dim2,
                 specfem::element::medium_tag::acoustic> {

public:
  constexpr static int dimension() { return 2; }

  constexpr static int components() { return 1; }
};

const std::string to_string(const medium_tag &medium,
                            const property_tag &property_tag);

const std::string to_string(const medium_tag &medium);

} // namespace element
} // namespace specfem
