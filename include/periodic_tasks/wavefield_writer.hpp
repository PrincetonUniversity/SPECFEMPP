#pragma once
#include "io/wavefield/writer.hpp"
#include "periodic_task.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace periodic_tasks {
/**
 * @brief Base plotter class
 *
 */
template <typename OutputLibrary>
class wavefield_writer : public periodic_task,
                         specfem::io::wavefield_writer<OutputLibrary> {
public:
  wavefield_writer(const std::string output_folder, const int time_interval,
                   const bool include_last_step)
      : periodic_task(time_interval, include_last_step),
        specfem::io::wavefield_writer<OutputLibrary>(output_folder) {}

  /**
   * @brief Check for keyboard interrupt and more, when running from Python
   *
   */
  void run(specfem::compute::assembly &assembly, const int istep) override {
    std::cout << "Writing wavefield files:" << std::endl;
    std::cout << "-------------------------------" << std::endl;
    this->set_istep(istep);
    this->write(assembly);
  }
};

} // namespace periodic_tasks
} // namespace specfem
