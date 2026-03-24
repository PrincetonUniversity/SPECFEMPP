#pragma once

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
