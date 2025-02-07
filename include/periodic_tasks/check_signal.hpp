#pragma once
#include "periodic_task.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace specfem {
namespace periodic_tasks {
/**
 * @brief Base plotter class
 *
 */
class check_signal : public periodic_task {
  using periodic_task::periodic_task;

  /**
   * @brief Plot the wavefield
   *
   */
  void run() override {
    if (PyErr_CheckSignals() != 0) {
      throw pybind11::error_already_set();
    }
  }
};

} // namespace periodic_tasks
} // namespace specfem
