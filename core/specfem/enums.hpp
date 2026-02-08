#pragma once

#include "enums/display.hpp"
#include "enums/wavefield.hpp"
#include "specfem/simulation.hpp"

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
 * @brief Output format of seismogram enumeration
 *
 */
enum class seismogram_format { seismic_unix, ascii };

} // namespace enums
} // namespace specfem
