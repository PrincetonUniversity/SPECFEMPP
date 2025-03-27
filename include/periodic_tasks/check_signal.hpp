#pragma once
#include "periodic_task.hpp"
#include <stdexcept>
#include <csignal>
#include <string>
#include <iostream>
#include <Kokkos_Core.hpp>

/**
 * @brief Catch signals
 * 
 * This function is catching the keyboard interrupt signal and other interrupt 
 * signals to parse them as exceptions to Python without crashing or hanging
 * the program. This is to avoid the program to hang via 
 * @code
 * if (PyErr_CheckSignals() != 0) {
 *     throw pybind11::error_already_set();
 * }
 * @endcode
 * if releasing the GIL in execute() function.
 */
void catch_signals() {
  auto handler = [](int code) { 
    std::string message =  "SIGNAL " + std::to_string(code);
    throw std::runtime_error(message);
  };
  signal(SIGINT, handler);
}

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
    // Catch signals
    catch_signals();
  }
};

} // namespace periodic_tasks
} // namespace specfem
