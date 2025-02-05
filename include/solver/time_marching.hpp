#pragma once

#include "coupled_interface/coupled_interface.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/simulation.hpp"
#include "enumerations/wavefield.hpp"
#include "kokkos_kernels/domain_kernels.hpp"
#include "kokkos_kernels/frechet_kernels.hpp"
#include "periodic_tasks/periodic_task.hpp"
#include "solver.hpp"
#include "timescheme/newmark.hpp"
#include "timescheme/timescheme.hpp"

namespace specfem {
namespace solver {
/**
 * @brief Time marching solver
 *
 * @tparam Simulation Type of the simulation (forward or combined)
 * @tparam DimensionType Dimension of the simulation (2D or 3D)
 * @tparam qp_type Quadrature points type defining compile time or runtime
 * quadrature points
 */
template <specfem::simulation::type Simulation,
          specfem::dimension::type DimensionType, int NGLL>
class time_marching;

/**
 * @brief Time marching solver for forward simulation
 */
template <specfem::dimension::type DimensionType, int NGLL>
class time_marching<specfem::simulation::type::forward, DimensionType, NGLL>
    : public solver {
public:
  /**
   * @name Constructors
   *
   */
  ///@{

  /**
   * @brief Construct a new time marching solver
   *
   * @param kernels Computational kernels
   * @param time_scheme Time scheme
   */
  time_marching(
      const specfem::kokkos_kernels::domain_kernels<
          specfem::wavefield::simulation_field::forward, DimensionType, NGLL>
          &kernels,
      const std::shared_ptr<specfem::time_scheme::time_scheme> time_scheme,
      const std::vector<
          std::shared_ptr<specfem::periodic_tasks::periodic_task> > &tasks)
      : kernels(kernels), time_scheme(time_scheme), tasks(tasks) {}

  ///@}

  /**
   * @brief Run the time marching solver
   */
  void run() override;

private:
  specfem::kokkos_kernels::domain_kernels<
      specfem::wavefield::simulation_field::forward, DimensionType,
      NGLL>
      kernels; ///< Computational kernels
  std::shared_ptr<specfem::time_scheme::time_scheme> time_scheme; ///< Time
                                                                  ///< scheme
  std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task> >
      tasks; ///< Periodic tasks
             ///< objects
};

/**
 * @brief Time marching solver for combined adjoint and backward simulations
 */
template <specfem::dimension::type DimensionType, int NGLL>
class time_marching<specfem::simulation::type::combined, DimensionType, NGLL>
    : public solver {
public:
  /**
   * @name Constructors
   *
   */
  ///@{

  /**
   * @brief Construct a new time marching solver
   *
   * @param assembly Spectral element assembly object
   * @param adjoint_kernels Adjoint computational kernels
   * @param backward_kernels Backward computational kernels
   * @param time_scheme Time scheme
   */
  time_marching(
      const specfem::compute::assembly &assembly,
      const specfem::kokkos_kernels::domain_kernels<
          specfem::wavefield::simulation_field::adjoint, DimensionType, NGLL>
          &adjoint_kernels,
      const specfem::kokkos_kernels::domain_kernels<
          specfem::wavefield::simulation_field::backward, DimensionType, NGLL>
          &backward_kernels,
      const std::shared_ptr<specfem::time_scheme::time_scheme> time_scheme,
      const std::vector<
          std::shared_ptr<specfem::periodic_tasks::periodic_task> > &tasks)
      : assembly(assembly), adjoint_kernels(adjoint_kernels),
        frechet_kernels(assembly), backward_kernels(backward_kernels),
        time_scheme(time_scheme), tasks(tasks) {}
  ///@}

  /**
   * @brief
   *
   */
  void run() override;

private:
  specfem::kokkos_kernels::domain_kernels<
      specfem::wavefield::simulation_field::adjoint, DimensionType,
      NGLL>
      adjoint_kernels; ///< Adjoint computational kernels
  specfem::kokkos_kernels::domain_kernels<
      specfem::wavefield::simulation_field::backward, DimensionType,
      NGLL>
      backward_kernels; ///< Backward computational kernels
  specfem::kokkos_kernels::frechet_kernels<DimensionType, NGLL>
      frechet_kernels;                 ///< Misfit kernels
  specfem::compute::assembly assembly; ///< Spectral element assembly object
  std::shared_ptr<specfem::time_scheme::time_scheme> time_scheme; ///< Time
                                                                  ///< scheme
  std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task> >
      tasks; ///< Periodic tasks
             ///< objects
};
} // namespace solver
} // namespace specfem
