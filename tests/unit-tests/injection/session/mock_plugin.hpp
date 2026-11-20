#pragma once

#include "specfem/plugin/injection_abi.h"
#include <cstdint>
#include <string>
#include <vector>

namespace specfem::injection_session_test {

/**
 * @brief A point group as received by the mock plugin's setup.
 */
struct MockGroup {
  specfem_medium medium = 0;
  specfem_property property = 0;
  std::int64_t begin = 0;
  std::int64_t count = 0;
  int ncomponents = 0;
  std::vector<std::string> property_names;
  std::vector<double> properties;
  std::vector<specfem_quantity_slot> slots;
};

/**
 * @brief Calls observed by the mock plugin since the last reset.
 */
struct MockCalls {
  int create = 0;
  int setup = 0;
  int fill = 0;
  int destroy = 0;
  std::int64_t setup_points = -1;      ///< Points received by the last setup
  std::vector<MockGroup> groups;       ///< Groups received by the last setup
  std::vector<int> fill_starts;        ///< first_step of every fill
  specfem_simulation_info info{};      ///< Info received by the last create
  specfem_communicator communicator{}; ///< Communicator received by create
};

/** @brief Calls observed so far. */
MockCalls &mock_calls();

/** @brief Forget all observed calls. */
void reset_mock_calls();

/**
 * @brief Register the mock plugin (once) and return its specification.
 *
 * The mock fills value(row, slot, step) = x[row] + 10 * slot + 1000 * step,
 * where row is the point's row across all groups. It supports
 * potential_rate on acoustic points and every other quantity on elastic
 * points, for any property. Its configuration path selects failures:
 * "fail-create", "fail-setup", "fail-fill", "fail-fill-rank0", "nan-fill",
 * and "no-traction" (supports neither traction nor potential_rate).
 *
 * @return "static:session-mock"
 */
const char *mock_plugin_spec();

/** @brief Value the mock writes for a row, slot, and absolute step. */
double mock_value(double x, int slot, int step);

} // namespace specfem::injection_session_test
