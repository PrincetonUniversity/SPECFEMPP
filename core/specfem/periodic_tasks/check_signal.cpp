#include "check_signal.hpp"
#include <signal.h>

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

// Global flag for signal state - must be volatile sig_atomic_t for thread
// safety
volatile sig_atomic_t signal_received = 0;

// Static function to handle signals
static void signal_handler(int code) {
  // Just set the flag and return
  signal_received = code;
  std::cout << "Signal " << code << " received (probably Ctrl-C)" << std::endl;
}

void catch_signals() {
  // Register our handler for SIGINT (Ctrl-C)
  signal(SIGINT, signal_handler);
}

template <specfem::dimension::type DimensionTag>
void specfem::periodic_tasks::check_signal<DimensionTag>::run(
    specfem::assembly::assembly<DimensionTag> &assembly, const int istep) {
  // Catch signals
  catch_signals();

  // Check if a signal was received
  if (signal_received) {
    // Handle the signal (e.g., throw an exception)
    std::string error_message =
        "Signal " + std::to_string(signal_received) + " received. Exiting...";
    // Reset the signal flag
    signal_received = 0;
    throw std::runtime_error(error_message);
  }
}

// Explicit template instantiations
template class specfem::periodic_tasks::check_signal<
    specfem::dimension::type::dim2>;
template class specfem::periodic_tasks::check_signal<
    specfem::dimension::type::dim3>;
