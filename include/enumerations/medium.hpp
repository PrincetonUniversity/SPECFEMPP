#pragma once

#include "enumerations/dimension.hpp"
#include <array>
#include <tuple>

namespace specfem {
namespace element {

/// See below how this is used within assembly.
constexpr int ntypes = 3; ///< Number of element types

// TODO: Since compute fields converts these enumerations into ints, we need to
// make sure that the order of the enumerations is such that any tag that is not
// an element in our domain is the last. This is hack and needs to be fixed in
// the future.

/**
 * @brief Medium tag enumeration
 *
 */
enum class medium_tag {
  elastic_sv,
  elastic_sh,
  acoustic,
  elastic,
  poroelastic
};

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

template <specfem::dimension::type Dimension,
          specfem::element::medium_tag MediumTag>
class attributes;

template <>
class attributes<specfem::dimension::type::dim2,
                 specfem::element::medium_tag::elastic_sv> {

public:
  constexpr static int dimension() { return 2; }

  constexpr static int components() { return 2; }
};

template <>
class attributes<specfem::dimension::type::dim2,
                 specfem::element::medium_tag::elastic_sh> {

public:
  constexpr static int dimension() { return 2; }

  constexpr static int components() { return 1; }
};

template <>
class attributes<specfem::dimension::type::dim2,
                 specfem::element::medium_tag::acoustic> {

public:
  constexpr static int dimension() { return 2; }

  constexpr static int components() { return 1; }
};

const std::string to_string(const medium_tag &medium,
                            const property_tag &property_tag,
                            const boundary_tag &boundary_tag);

const std::string to_string(const medium_tag &medium,
                            const property_tag &property_tag);

const std::string to_string(const medium_tag &medium);

} // namespace element
} // namespace specfem
