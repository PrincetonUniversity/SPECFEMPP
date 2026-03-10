#pragma once

#include "specfem/element.hpp"
#include "specfem/element_connections.hpp"

/**
 * @brief Element coupling configuration for multi-physics interfaces.
 *
 * Provides compile-time interface configuration for coupling different
 * physics media (elastic-acoustic, acoustic-elastic). Defines coupling
 * directions, flux schemes, and field type resolution through template
 * specializations.
 *
 */
namespace specfem::element_coupling {}

namespace specfem::point {

template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag, bool UseSIMD>
struct acceleration;

template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag, bool UseSIMD>
struct displacement;

} // namespace specfem::point

namespace specfem::element_coupling {

/**
 * @brief Interface coupling direction types.
 *
 * Directional coupling: elastic_acoustic (elastic→acoustic),
 * acoustic_elastic (acoustic→elastic).
 */
enum class interface_tag {
  elastic_acoustic, ///< Elastic to acoustic interface - elastic field couples
                    ///< to acoustic
  acoustic_elastic  ///< Acoustic to elastic interface - acoustic field couples
                    ///< to elastic
};

/**
 * @brief Flux scheme used for a coupling
 */
enum class flux_scheme_tag {
  natural, ///< Original SPECFEM acoustic-elastic interface (Komatitsch et al.
           ///< 2000)
  symmetric_interior_penalty ///< SIPG (Grote et al., Riviere et al., Antonietti
                             ///< et al., etc.)
};

} // namespace specfem::element_coupling

#include "element_coupling/attributes.hpp"
#include "element_coupling/to_string.hpp"
