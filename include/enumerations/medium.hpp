#pragma once

#include "enumerations/dimension.hpp"

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
enum class property_tag { isotropic };

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
