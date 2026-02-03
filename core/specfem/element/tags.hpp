#pragma once

namespace specfem {
namespace element {

/**
 * @brief Element medium types for physics simulations.
 *
 * Defines wave propagation physics: elastic (P/SV/SH waves), acoustic (pressure
 * waves), poroelastic (fluid-solid interaction), and electromagnetic (TE/TM
 * modes).
 */
enum class medium_tag {
  elastic_psv,        ///< 2D elastic medium with P and SV waves
  elastic_sh,         ///< 2D elastic medium with SH waves
  elastic_psv_t,      ///< 2D elastic PSV with transverse spin (Cosserat)
  acoustic,           ///< Acoustic medium (pressure waves)
  poroelastic,        ///< Poroelastic medium (Biot theory)
  electromagnetic_te, ///< 2D electromagnetic TE modes
  elastic,            ///< 3D elastic medium (full displacement field)
  elastic_spin,       ///< Elastic medium with spin dynamics
  electromagnetic,    ///< Electromagnetic medium (TE and TM modes)
};

/**
 * @brief Material property symmetries.
 *
 * Controls material tensor structure: isotropic (scalar properties),
 * anisotropic (full tensor), isotropic_cosserat (with microrotation).
 */
enum class property_tag {
  isotropic,         ///< Isotropic material (scalar properties)
  anisotropic,       ///< Anisotropic material (full tensor)
  isotropic_cosserat ///< Isotropic Cosserat material (with microrotation)
};

/**
 * @brief Boundary condition types for domain edges.
 *
 * Defines how waves interact with domain boundaries: free surfaces,
 * absorbing conditions (Stacey), and composite boundary treatments.
 */
enum class boundary_tag {
  // primary boundaries
  none,                  ///< No boundary condition
  acoustic_free_surface, ///< Acoustic free surface (zero pressure)
  stacey,                ///< Stacey absorbing boundary condition

  // composite boundaries
  composite_stacey_dirichlet ///< Combined Stacey-Dirichlet boundary
};

enum class attenuation_tag {
  none,               ///< No attenuation
  constant_isotropic, ///< Constant Q-Band attenuation
};

} // namespace element
} // namespace specfem
