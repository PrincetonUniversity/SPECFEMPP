#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/simulation.hpp"
#include "enumerations/wavefield.hpp"
#include "solver.hpp"
#include "specfem/kokkos_kernels.hpp"
#include "specfem/periodic_tasks.hpp"
#include "specfem/timescheme.hpp"

namespace specfem {
namespace solver {
/**
 * @brief Explicit time-stepping solver for spectral element wave propagation
 *
 * Implements predictor-corrector time integration schemes for various wave
 * types (acoustic, elastic, poroelastic). Handles multi-physics coupling,
 * source interactions, and seismogram generation.
 *
 * @tparam Simulation Simulation type (forward or combined adjoint+backward)
 * @tparam DimensionTag Spatial dimension (2D or 3D)
 * @tparam NGLL Number of Gauss-Lobatto-Legendre quadrature points per element
 */
template <specfem::simulation::type Simulation,
          specfem::dimension::type DimensionTag, int NGLL>
class time_marching;

/**
 * @brief Time marching solver for forward simulation
 */
template <specfem::dimension::type DimensionTag, int NGLL>
class time_marching<specfem::simulation::type::forward, DimensionTag, NGLL>
    : public solver {
public:
  constexpr static auto dimension_tag =
      DimensionTag; ///< Dimension of the problem
  /**
   * @name Constructors
   *
   */
  ///@{

  /**
   * @brief Construct solver for forward wave propagation
   *
   * @param kernels Domain computational kernels for wavefield updates
   * @param time_scheme Time integration scheme (e.g., Newmark)
   * @param tasks Periodic tasks executed during simulation (e.g., output,
   * plotting)
   * @param assembly Spectral element assembly containing mesh and field data
   */
  time_marching(
      const specfem::kokkos_kernels::domain_kernels<
          specfem::wavefield::simulation_field::forward, DimensionTag, NGLL>
          &kernels,
      const std::shared_ptr<specfem::time_scheme::time_scheme> time_scheme,
      const std::vector<std::shared_ptr<
          specfem::periodic_tasks::periodic_task<DimensionTag> > > &tasks,
      specfem::assembly::assembly<dimension_tag> assembly)
      : kernels(kernels), time_scheme(time_scheme), tasks(tasks),
        assembly(assembly) {}

  ///@}

  /**
   * @brief Execute time-stepping loop for forward simulation
   *
   * Performs explicit time integration using predictor-corrector phases:
   * 1. Predictor: updates velocity/displacement from @f$ t^n @f$ to @f$
   * t^{n+1/2} @f$
   * 2. Corrector: finalizes update to @f$ t^{n+1} @f$
   *
   * @par Execution Order (critical for multi-physics coupling):
   * At each timestep, media are processed in this specific sequence:
   * 1. **Predictor phase** for all media (acoustic, elastic, poroelastic)
   * 2. **Acoustic update**: wavefield computation → corrector phase
   * 3. **Elastic update**: wavefield computation (elastic, elastic_psv,
   * elastic_sh) → corrector phase
   * 4. **Poroelastic update**: wavefield computation → corrector phase
   *
   * This ordering ensures proper coupling at fluid-solid and solid-solid
   * interfaces. Computes seismograms and runs periodic tasks at specified
   * intervals.
   */
  void run() override;

private:
  specfem::kokkos_kernels::domain_kernels<
      specfem::wavefield::simulation_field::forward, DimensionTag,
      NGLL>
      kernels; ///< Computational kernels
  std::shared_ptr<specfem::time_scheme::time_scheme> time_scheme; ///< Time
                                                                  ///< scheme
  std::vector<
      std::shared_ptr<specfem::periodic_tasks::periodic_task<DimensionTag> > >
      tasks; ///< Periodic tasks
  ///< objects
  specfem::assembly::assembly<dimension_tag> assembly; ///< Spectral element
                                                       ///< assembly object
};

/**
 * @brief Time marching solver for combined adjoint and backward simulations
 */
template <specfem::dimension::type DimensionTag, int NGLL>
class time_marching<specfem::simulation::type::combined, DimensionTag, NGLL>
    : public solver {
public:
  constexpr static auto dimension_tag =
      DimensionTag; ///< Dimension of the problem
  /**
   * @name Constructors
   *
   */
  ///@{

  /**
   * @brief Construct solver for combined adjoint and backward simulations
   *
   * Used for computing Fréchet derivatives (sensitivity kernels) via the
   * adjoint method.
   *
   * @param assembly Spectral element assembly containing mesh and field data
   * @param adjoint_kernels Domain kernels for adjoint wavefield propagation
   * @param backward_kernels Domain kernels for backward wavefield propagation
   * @param time_scheme Time integration scheme
   * @param tasks Periodic tasks executed during simulation
   */
  time_marching(
      const specfem::assembly::assembly<dimension_tag> &assembly,
      const specfem::kokkos_kernels::domain_kernels<
          specfem::wavefield::simulation_field::adjoint, DimensionTag, NGLL>
          &adjoint_kernels,
      const specfem::kokkos_kernels::domain_kernels<
          specfem::wavefield::simulation_field::backward, DimensionTag, NGLL>
          &backward_kernels,
      const std::shared_ptr<specfem::time_scheme::time_scheme> time_scheme,
      const std::vector<std::shared_ptr<
          specfem::periodic_tasks::periodic_task<dimension_tag> > > &tasks)
      : assembly(assembly), adjoint_kernels(adjoint_kernels),
        frechet_kernels(assembly), backward_kernels(backward_kernels),
        time_scheme(time_scheme), tasks(tasks) {}
  ///@}

  /**
   * @brief Execute time-stepping loop for combined adjoint and backward
   * simulations
   *
   * Performs backward time integration of both adjoint and backward wavefields:
   * - Adjoint field: propagates adjoint sources backward in time
   * - Backward field: reconstructs forward wavefield from stored buffer
   *
   * @par Execution Order (per timestep, backward iteration):
   * **Adjoint wavefield (forward-style update):**
   * 1. Predictor phase for all media
   * 2. Acoustic update → corrector phase
   * 3. Elastic update → corrector phase
   * 4. Poroelastic update → corrector phase
   *
   * **Backward wavefield (reverse-time reconstruction):**
   * 1. Predictor phase for all media
   * 2. Elastic update → corrector phase
   * 3. Acoustic update → corrector phase
   * 4. Poroelastic update → corrector phase
   *
   * **Fréchet kernels:** Computed after both wavefields are updated,
   * correlating adjoint and backward fields for gradient-based inversion.
   */
  void run() override;

private:
  specfem::kokkos_kernels::domain_kernels<
      specfem::wavefield::simulation_field::adjoint, DimensionTag,
      NGLL>
      adjoint_kernels; ///< Adjoint computational kernels
  specfem::kokkos_kernels::domain_kernels<
      specfem::wavefield::simulation_field::backward, DimensionTag,
      NGLL>
      backward_kernels; ///< Backward computational kernels
  specfem::kokkos_kernels::frechet_kernels<DimensionTag, NGLL>
      frechet_kernels;                                 ///< Misfit kernels
  specfem::assembly::assembly<dimension_tag> assembly; ///< Spectral element
                                                       ///< assembly object
  std::shared_ptr<specfem::time_scheme::time_scheme> time_scheme; ///< Time
                                                                  ///< scheme
  std::vector<
      std::shared_ptr<specfem::periodic_tasks::periodic_task<dimension_tag> > >
      tasks; ///< Periodic tasks
             ///< objects
};
} // namespace solver
} // namespace specfem
