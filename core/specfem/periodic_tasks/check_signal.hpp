#pragma once
#include "enumerations/interface.hpp"
#include "periodic_task.hpp"
#include <Kokkos_Core.hpp>
#include <csignal>
#include <iostream>
#include <stdexcept>
#include <string>

namespace specfem {
namespace periodic_tasks {
/**
 * @brief Base plotter class
 *
 */
template <specfem::dimension::type DimensionTag>
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
