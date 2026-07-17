#pragma once

#include "mpi_buffers.hpp"
#include "solver.hpp"
#include "specfem/compute.hpp"
#include "specfem/element.hpp"
#include "specfem/enums.hpp"
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
 * @tparam Simulation Simulation type (forward, combined, or combined_undoatt)
 * @tparam DimensionTag Spatial dimension (2D or 3D)
 * @tparam NGLL Number of Gauss-Lobatto-Legendre quadrature points per element
 */
template <specfem::simulation::type Simulation,
          specfem::element::dimension_tag DimensionTag, int NGLL>
class time_marching;

/**
 * @brief Time marching solver for forward simulation
 */
template <specfem::element::dimension_tag DimensionTag, int NGLL>
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
   * @param time_scheme Time integration scheme (e.g., Newmark)
   * @param tasks Periodic tasks executed during simulation (e.g., output,
   * plotting)
   * @param assembly Spectral element assembly containing mesh and field data
   */
  time_marching(
      const std::shared_ptr<specfem::time_scheme::time_scheme> time_scheme,
      const std::vector<
          std::shared_ptr<specfem::periodic_tasks::periodic_task<DimensionTag>>>
          &tasks,
      specfem::assembly::assembly<dimension_tag> assembly)
      : time_scheme(time_scheme), tasks(tasks), assembly(assembly),
        mpi_buffers(specfem::solver::make_mpi_buffers(this->assembly)) {}

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
  std::shared_ptr<specfem::time_scheme::time_scheme> time_scheme; ///< Time
                                                                  ///< scheme
  std::vector<
      std::shared_ptr<specfem::periodic_tasks::periodic_task<DimensionTag>>>
      tasks; ///< Periodic tasks
  ///< objects
  specfem::assembly::assembly<dimension_tag> assembly; ///< Spectral element
                                                       ///< assembly object
  specfem::solver::MPIBuffers<dimension_tag>
      mpi_buffers; ///< Solver-owned MPI buffers for acceleration and
                   ///< mass-matrix exchange (inner/outer overlap)
};

/**
 * @brief Time marching solver for combined adjoint and backward simulations
 */
template <specfem::element::dimension_tag DimensionTag, int NGLL>
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
   * @param time_scheme Time integration scheme
   * @param tasks Periodic tasks executed during simulation
   */
  time_marching(
      const specfem::assembly::assembly<dimension_tag> &assembly,
      const std::shared_ptr<specfem::time_scheme::time_scheme> time_scheme,
      const std::vector<std::shared_ptr<
          specfem::periodic_tasks::periodic_task<dimension_tag>>> &tasks)
      : assembly(assembly), time_scheme(time_scheme), tasks(tasks),
        mpi_buffers(specfem::solver::make_mpi_buffers(this->assembly)) {}
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
  specfem::assembly::assembly<dimension_tag> assembly; ///< Spectral element
                                                       ///< assembly object
  std::shared_ptr<specfem::time_scheme::time_scheme> time_scheme; ///< Time
                                                                  ///< scheme
  std::vector<
      std::shared_ptr<specfem::periodic_tasks::periodic_task<dimension_tag>>>
      tasks; ///< Periodic tasks
             ///< objects
  specfem::solver::MPIBuffers<dimension_tag>
      mpi_buffers; ///< Solver-owned MPI buffers for acceleration and
                   ///< mass-matrix exchange (inner/outer overlap)
};
/**
 * @brief Time marching solver for adjoint simulations with exact forward replay
 *
 * Implements the UNDO_ATTENUATION strategy from Komatitsch et al. (2016).
 * A prior forward simulation must checkpoint wavefield snapshots
 * (displacement, velocity, acceleration, attenuation memory variables) to disk
 * at every NT_DUMP_ATTENUATION steps. The adjoint simulation iterates backward
 * over time subsets. For each subset the checkpoint is loaded from disk, the
 * forward wavefield is replayed exactly (including attenuation physics) over
 * the subset window, and the resulting displacements are stored in a RAM
 * buffer. The adjoint wavefield then sweeps through the same window
 * (reversed), reading reconstructed forward displacements from the RAM buffer
 * and accumulating Fréchet kernels.
 *
 * Because the forward wavefield is replayed from disk snapshots (not reversed
 * in time), attenuation memory variables are always in the correct causal
 * forward state when kernels are computed. Strain is recomputed on-the-fly
 * from the reconstructed displacement, matching the Fortran
 * UNDO_ATTENUATION_AND_OR_PML implementation exactly.
 */
template <specfem::element::dimension_tag DimensionTag, int NGLL>
class time_marching<specfem::simulation::type::combined_undoatt, DimensionTag,
                    NGLL> : public solver {
public:
  constexpr static auto dimension_tag =
      DimensionTag; ///< Dimension of the problem

  /**
   * @brief Construct solver for undo-attenuation adjoint simulation
   *
   * @param assembly     Spectral element assembly
   * @param time_scheme  Time integration scheme
   * @param tasks        Periodic tasks (must include a wavefield_reader
   *                     configured to read checkpoints written by the forward
   *                     run every NT_DUMP_ATTENUATION steps)
   */
  time_marching(
      const specfem::assembly::assembly<dimension_tag> &assembly,
      const std::shared_ptr<specfem::time_scheme::time_scheme> time_scheme,
      const std::vector<std::shared_ptr<
          specfem::periodic_tasks::periodic_task<dimension_tag>>> &tasks,
      int checkpoint_buffer_subdivisions = 1)
      : assembly(assembly), time_scheme(time_scheme), tasks(tasks),
        mpi_buffers(specfem::solver::make_mpi_buffers(this->assembly)),
        checkpoint_buffer_subdivisions(checkpoint_buffer_subdivisions) {}

  /**
   * @brief Execute adjoint pass with windowed forward replay.
   */
  void run() override;

private:
  specfem::assembly::assembly<dimension_tag> assembly;
  std::shared_ptr<specfem::time_scheme::time_scheme> time_scheme;
  std::vector<
      std::shared_ptr<specfem::periodic_tasks::periodic_task<dimension_tag>>>
      tasks;
  specfem::solver::MPIBuffers<dimension_tag> mpi_buffers;
  int checkpoint_buffer_subdivisions;
};

} // namespace solver
} // namespace specfem
