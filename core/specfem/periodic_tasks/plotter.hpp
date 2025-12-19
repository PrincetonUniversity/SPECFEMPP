#pragma once
#include "periodic_task.hpp"

namespace specfem {
namespace periodic_tasks {
/**
 * @brief Base plotter class
 *
 */
template <specfem::dimension::type DimensionTag>
class plotter : public periodic_task<DimensionTag> {
  using periodic_task<DimensionTag>::periodic_task;
};

} // namespace periodic_tasks
} // namespace specfem
