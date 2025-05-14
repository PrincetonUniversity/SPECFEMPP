#pragma once
#include "io/operators.hpp"
#include "io/wavefield/reader.hpp"
#include "periodic_task.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace periodic_tasks {
/**
 * @brief Base plotter class
 *
 */
template <template <typename OpType> class IOLibrary>
class wavefield_reader : public periodic_task {
private:
  specfem::io::wavefield_reader<IOLibrary<specfem::io::read> > reader;

public:
  wavefield_reader(const std::string output_folder, const int time_interval,
                   const bool include_last_step)
      : periodic_task(time_interval, include_last_step),
        reader(specfem::io::wavefield_reader<IOLibrary<specfem::io::read> >(
            output_folder)) {}

  /**
   * @brief Check for keyboard interrupt and more, when running from Python
   *
   */
  void run(specfem::compute::assembly &assembly, const int istep) override {
    std::cout << "Reading wavefield files:" << std::endl;
    std::cout << "-------------------------------" << std::endl;
    reader.read(assembly, istep);
  }
};

} // namespace periodic_tasks
} // namespace specfem
