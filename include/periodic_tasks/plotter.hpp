#pragma once
#include "periodic_task.hpp"

namespace specfem {
namespace periodic_tasks {
/**
 * @brief Base plotter class
 *
 */
class plotter : public periodic_task {
  using periodic_task::periodic_task;
};

} // namespace periodic_tasks
} // namespace specfem
