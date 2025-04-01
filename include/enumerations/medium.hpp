#pragma once

#include "enumerations/dimension.hpp"
#include <array>
#include <tuple>

namespace specfem {
namespace element {

/// See below how this is used within assembly.
constexpr int ntypes = 6; ///< Number of element types

// TODO: Since compute fields converts these enumerations into ints, we need to
// make sure that the order of the enumerations is such that any tag that is not
// an element in our domain is the last. This is hack and needs to be fixed in
// the future.

/**
 * @brief Medium tag enumeration
 *
 * This enumeration is used to identify the type of medium in the element.
 * There are several types of media, including:
 * - elastic_psv: 2-D elastic medium with P and SV waves
 * - elastic_sh: 2-D elastic medium with SH waves
 * - elastic_psv_t: 2-D elastic medium with P and SV waves and a transverse
 *                  spin vector
 * - acoustic: 2-D acoustic medium
 * - poroelastic: 2-D poroelastic medium only supporting P and SV waves
 * - electromagnetic_te: 2-D electromagnetic medium with transverse electric
 *                       modes
 * - electromagnetic: 2-D electromagnetic medium
 *                    for capturing the electromagnetic parameters for TE and
 *                    TM systems
 * - elastic: 2-D elastic medium for capturing the elastic parameters for
 *   PSV and SH systems
 *
 */
enum class medium_tag {
  elastic_psv,
  elastic_sh,
  elastic_psv_t,
  acoustic,
  poroelastic,
  electromagnetic_te,
  elastic,
  elastic_spin,
  electromagnetic,
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
                 specfem::element::medium_tag::elastic_psv_t> {

public:
  constexpr static int dimension() { return 2; }

  constexpr static int components() { return 3; }
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

const std::string to_string(const property_tag &property);

const std::string to_string(const boundary_tag &boundary);

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

template <specfem::element::medium_tag MediumTag>
using is_elastic_spin = typename std::conditional_t<
    (MediumTag == specfem::element::medium_tag::elastic_spin ||
     MediumTag == specfem::element::medium_tag::elastic_psv_t),
    std::true_type, std::false_type>::type;
} // namespace element
} // namespace specfem
