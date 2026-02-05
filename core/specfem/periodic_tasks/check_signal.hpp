#pragma once
#include "periodic_task.hpp"
#include "specfem/enums.hpp"
#include <Kokkos_Core.hpp>
#include <csignal>
#include <iostream>
#include <stdexcept>
#include <string>

namespace specfem {
namespace periodic_tasks {
/**
 * @brief Signal checker class for handling interrupts during simulation
 *
 */
template <specfem::element::dimension_tag DimensionTag>
class check_signal : public periodic_task<DimensionTag> {
  using periodic_task<DimensionTag>::periodic_task;

  /**
   * @brief Check for keyboard interrupt and more, when running from Python
   *
   */
  void run(specfem::assembly::assembly<DimensionTag> &assembly,
           const int istep) override;
};

} // namespace periodic_tasks
} // namespace specfem
