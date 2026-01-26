#pragma once

/**
 * @brief Physics computations for seismic wave propagation media.
 *
 * Provides computational functions for acoustic, elastic, poroelastic,
 * electromagnetic, and cosserat media in 2D/3D with isotropic/anisotropic
 * properties. Uses template dispatch for compile-time medium selection.
 *
 * **Core functions:**
 * - `compute_stress()`: Stress tensor from field derivatives
 * - `compute_wavefield()`: Wavefield from intrinsic fields
 * - `compute_source_contribution()`: Source terms
 * - `compute_frechet_derivatives()`: Sensitivity kernels
 * - `material<>`, `domain_properties<>`, `domain_kernels<>`: Material property
 * management
 */
namespace specfem::medium {}

#include "medium_physics/compute_cosserat_couple_stress.hpp"
#include "medium_physics/compute_cosserat_stress.hpp"
#include "medium_physics/compute_coupling.hpp"
#include "medium_physics/compute_damping_force.hpp"
#include "medium_physics/compute_frechet_derivatives.hpp"
#include "medium_physics/compute_mass_matrix.hpp"
#include "medium_physics/compute_source.hpp"
#include "medium_physics/compute_stress.hpp"
#include "medium_physics/compute_wavefield.hpp"
