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
class check_signal : public periodic_task {
  using periodic_task::periodic_task;

  /**
   * @brief Check for keyboard interrupt and more, when running from Python
   *
   */
  void
  run(specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly,
      const int istep) override;
};

} // namespace periodic_tasks
} // namespace specfem
