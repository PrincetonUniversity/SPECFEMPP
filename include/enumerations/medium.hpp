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

/*
 * @brief Default attributes class
 *
 * This class is used to define the default attributes of the element. It is
 * used to define the activation of extra physics, such as the damping force in
 * poroelasticity. This class is used as a base class for the attributes class.
 *
 */
struct default_attributes {
  constexpr static bool damping_force = false;
};

/*
 * @brief Attributes class
 *
 * This class is used to define the attributes of the element. It is specialized
 * for each combination of dimension and medium tag. The attributes are defined
 * in the specialization of the class. The attributes are used to define the
 * number of components, the dimension, and other extra physics, such as the
 * damping force in poroelasticity
 *
 * @tparam Dimension Dimension of the element
 * @tparam MediumTag Medium tag of the element
 *
 * @note The default attributes are defined in the default_attributes class.
 *
 * The default constructor of the attributes class throws an error if the
 * attributes are used without being specialized. This is used to ensure that
 * the attributes are always specialized for the combination of dimension and
 * medium tag.
 */
template <specfem::dimension::type Dimension,
          specfem::element::medium_tag MediumTag>
class attributes : public default_attributes {
  static_assert(sizeof(Dimension) == 0 || sizeof(MediumTag) == 0,
                "Unregistered attributes tag! Please add a specialization for "
                "dimension/medium enum value.");
};

template <>
class attributes<specfem::dimension::type::dim2,
                 specfem::element::medium_tag::elastic_psv>
    : public default_attributes {

public:
  constexpr static int dimension() { return 2; }

  constexpr static int components() { return 2; }
};

template <>
class attributes<specfem::dimension::type::dim2,
                 specfem::element::medium_tag::elastic_sh>
    : public default_attributes {

public:
  constexpr static int dimension() { return 2; }

  constexpr static int components() { return 1; }
};

template <>
class attributes<specfem::dimension::type::dim2,
                 specfem::element::medium_tag::acoustic>
    : public default_attributes {

public:
  constexpr static int dimension() { return 2; }

  constexpr static int components() { return 1; }
};

template <>
class attributes<specfem::dimension::type::dim2,

                 specfem::element::medium_tag::poroelastic>
    : public default_attributes {
public:
  constexpr static int dimension() { return 2; }

  constexpr static int components() { return 4; }

  constexpr static bool damping_force = true;
};

template <>
class attributes<specfem::dimension::type::dim2,
                 specfem::element::medium_tag::electromagnetic_te>
    : public default_attributes {
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

} // namespace element
} // namespace specfem
