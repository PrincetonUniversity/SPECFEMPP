#pragma once

#include "specfem/plugin/injection_abi.h"
#include <cstdint>
#include <vector>

namespace specfem::injection_session_test {

/**
 * @brief Calls observed by the mock plugin since the last reset.
 */
struct MockCalls {
  int create = 0;
  int setup = 0;
  int fill = 0;
  int destroy = 0;
  std::int64_t setup_points = -1;   ///< Points received by the last setup
  std::vector<int> fill_starts;     ///< first_step of every fill
  spfp_simulation_info info{};      ///< Info received by the last create
  spfp_communicator communicator{}; ///< Communicator received by create
};

/** @brief Calls observed so far. */
MockCalls &mock_calls();

/** @brief Forget all observed calls. */
void reset_mock_calls();

/**
 * @brief Register the mock plugin (once) and return its specification.
 *
 * The mock fills value(row, slot, step) = x[row] + 10 * slot + 1000 * step.
 * Its configuration path selects failures: "fail-create", "fail-setup",
 * "fail-fill", "fail-fill-rank0", "nan-fill", and "no-traction" (does not
 * support traction or potential_rate).
 *
 * @return "static:session-mock"
 */
const char *mock_plugin_spec();

/** @brief Value the mock writes for a row, slot, and absolute step. */
double mock_value(double x, int slot, int step);

} // namespace specfem::injection_session_test
