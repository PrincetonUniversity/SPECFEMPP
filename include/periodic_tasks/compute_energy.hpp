#pragma once

#include "compute/assembly/assembly.hpp"
#include "periodic_tasks/periodic_task.hpp"
#include "specfem_setup.hpp"

namespace specfem {
namespace periodic_tasks {

template <int NGLL> class compute_energy : public periodic_task {
public:
  constexpr static auto ngll = NGLL;
  using periodic_task::periodic_task;

  void run() override;

private:
  specfem::compute::assembly assembly;

  type_real compute() const;
};

} // namespace periodic_tasks
} // namespace specfem
