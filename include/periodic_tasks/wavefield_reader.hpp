#pragma once
#include "io/wavefield/reader.hpp"
#include "periodic_task.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace periodic_tasks {
/**
 * @brief Base plotter class
 *
 */
template <typename IOLibrary>
class wavefield_reader : public periodic_task,
                         specfem::io::wavefield_reader<IOLibrary> {
public:
  wavefield_reader(const std::string output_folder, const int time_interval,
                   const bool include_last_step)
      : periodic_task(time_interval, include_last_step),
        specfem::io::wavefield_reader<IOLibrary>(output_folder) {}

  /**
   * @brief Check for keyboard interrupt and more, when running from Python
   *
   */
  void run(specfem::compute::assembly &assembly, const int istep) override {
    std::cout << "Reading wavefield files:" << std::endl;
    std::cout << "-------------------------------" << std::endl;
    this->set_istep(istep);
    this->read(assembly);
  }
};

} // namespace periodic_tasks
} // namespace specfem
