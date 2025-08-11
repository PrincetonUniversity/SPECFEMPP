#pragma once

#include "kokkos_abstractions.h"
#include <Kokkos_Core.hpp>

namespace specfem {
/**
 * @namespace enums namespace is used to store enumerations.
 *
 */
namespace enums {

// Two types of elastic waves are possible in 2-D: PSV and SH
// PSV: P-SV wave (compressional and vertically polarized shear)
// SH: S-H wave (shear wave with horizontal polarization)
enum class elastic_wave { psv, sh };

// Two types of perdincularly polarized modes are possible for electromagnetic
// waves:
// TE: Transverse Electric mode
// TM: Transverse Magnetic mode
enum class electromagnetic_wave { te, tm };

// Two types of elastic spin systems are possible in 2-D: PSV-T and SH-VL
// PSV-T: P-SV wave with transverse spin vector
// SH-VL: SH wave with vertical and longitudinal spin vectors
enum class elastic_spin_wave { psv_t, sh_vl };

/**
 * @brief Cartesian axes enumeration
 *
 */
enum class axes { x, y, z };

namespace seismogram {

/**
 * @brief Output format of seismogram enumeration
 *
 */
enum format { seismic_unix, ascii };

} // namespace seismogram

/**
 * @namespace edge namespace is used to store enumerations used to describe the
 * edges
 *
 */
namespace edge {
/**
 * @brief Edge type enumeration
 *
 */
enum type {
  NONE = 0,   /// Not an edge
  TOP = 1,    ///< Top edge
  BOTTOM = 2, ///< Bottom edge
  LEFT = 3,   ///< Left edge
  RIGHT = 4   ///< Right edge
};

constexpr int num_edges = 5; ///< Number of edges in the mesh
} // namespace edge

namespace time_scheme {
/**
 * @brief Time scheme enumeration
 *
 */
enum class type {
  newmark, ///< Newmark time scheme
};

} // namespace time_scheme

} // namespace enums
} // namespace specfem
