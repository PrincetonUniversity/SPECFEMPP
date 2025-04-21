#pragma once

#include "enumerations/dimension.hpp"
#include <array>
#include <tuple>

namespace specfem {
namespace element {

/// See below how this is used within assembly.
constexpr int ntypes = 5; ///< Number of element types

// TODO: Since compute fields converts these enumerations into ints, we need to
// make sure that the order of the enumerations is such that any tag that is not
// an element in our domain is the last. This is hack and needs to be fixed in
// the future.

/**
 * @brief Medium tag enumeration
 *
 */
enum class medium_tag {
  elastic_psv,
  elastic_sh,
  acoustic,
  poroelastic,
  electromagnetic_te,
  electromagnetic,
  elastic,
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
                 specfem::element::medium_tag::elastic_psv> {

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

template <>
class attributes<specfem::dimension::type::dim2,

                 specfem::element::medium_tag::poroelastic> {
public:
  constexpr static int dimension() { return 2; }

  constexpr static int components() { return 4; }
};

template <>
class attributes<specfem::dimension::type::dim2,
                 specfem::element::medium_tag::electromagnetic_te> {
public:
  constexpr static int dimension() { return 2; }

  constexpr static int components() { return 2; }
};

const std::string to_string(const medium_tag &medium,
                            const property_tag &property_tag,
                            const boundary_tag &boundary_tag);

const std::string to_string(const medium_tag &medium,
                            const property_tag &property_tag);

const std::string to_string(const medium_tag &medium);

// template class to enable specialization for elastic media
template <specfem::element::medium_tag MediumTag>
using is_elastic = typename std::conditional_t<
    (MediumTag == specfem::element::medium_tag::elastic ||
     MediumTag == specfem::element::medium_tag::elastic_psv ||
     MediumTag == specfem::element::medium_tag::elastic_sh),
    std::true_type, std::false_type>::type;

template <specfem::element::medium_tag MediumTag>
using is_electromagnetic = typename std::conditional_t<
    (MediumTag == specfem::element::medium_tag::electromagnetic ||
     MediumTag == specfem::element::medium_tag::electromagnetic_te),
    std::true_type, std::false_type>::type;

// Has damping force
// Base dispatcher struct for tag validation
template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
struct damping_force {
  // Will trigger for unregistered tags
  static_assert(sizeof(Dimension) == 0,
                "Unregistered damping tag! Please add a specialization for "
                "medium enum value.");
  static_assert(sizeof(MediumTag) == 0,
                "Unregistered damping tag! Please add a specialization for "
                "medium enum value.");
  static_assert(sizeof(PropertyTag) == 0,
                "Unregistered damping tag! Please add a specialization for "
                "medium enum value.");
  static constexpr bool activated = false;
};

// Elastic PSV Isotropic (Not activated)
template <>
struct damping_force<specfem::element::medium_tag::elastic_psv,
                     specfem::element::property_tag::isotropic> {
  static constexpr bool activated = false;
};

// Elastic SH Isotropic (Not activated)
template <>
struct damping_force<specfem::element::medium_tag::elastic_sh,
                     specfem::element::property_tag::isotropic> {
  static constexpr bool activated = false;
};

// Elastic PSV Anisotropic (Not activated)
template <>
struct damping_force<specfem::element::medium_tag::elastic_psv,
                     specfem::element::property_tag::anisotropic> {
  static constexpr bool activated = false;
};
// Elastic SH Anisotropic (Not activated)
template <>
struct damping_force<specfem::element::medium_tag::elastic_sh,
                     specfem::element::property_tag::anisotropic> {
  static constexpr bool activated = false;
};
// Acoustic Isotropic (Not activated)
template <>
struct damping_force<specfem::element::medium_tag::acoustic,
                     specfem::element::property_tag::isotropic> {
  static constexpr bool activated = false;
};

// Poroelastic Isotropic (Activated)
template <>
struct damping_force<specfem::element::medium_tag::poroelastic,
                     specfem::element::property_tag::isotropic> {
  static constexpr bool activated = true;
};

} // namespace element
} // namespace specfem
