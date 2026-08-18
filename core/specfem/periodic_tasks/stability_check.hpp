#pragma once
#include "periodic_task.hpp"

namespace specfem {
namespace periodic_tasks {

/**
 * @brief Periodic divergence / stability check for the forward time loop.
 *
 * Every `check_interval` steps, computes the global max |displacement|
 * across all media and MPI ranks. Aborts if the value is non-finite or
 * exceeds `specfem::constants::STABILITY_THRESHOLD`.
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 */
template <specfem::element::dimension_tag DimensionTag>
class stability_check : public periodic_task<DimensionTag> {
public:
  /**
   * @brief Construct a stability check task.
   * @param check_interval Number of time steps between checks.
   */
  explicit stability_check(int check_interval)
      : periodic_task<DimensionTag>(check_interval) {}

  void run(specfem::assembly::assembly<DimensionTag> &assembly,
           const int istep) override;
};

} // namespace periodic_tasks
} // namespace specfem
