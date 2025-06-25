#pragma once
#include "io/operators.hpp"
#include "io/wavefield/writer.hpp"
#include "periodic_task.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace periodic_tasks {
/**
 * @brief Base plotter class
 *
 */
template <template <typename OpType> class IOLibrary>
class wavefield_writer : public periodic_task {
private:
  specfem::io::wavefield_writer<IOLibrary<specfem::io::write> > writer;

public:
  wavefield_writer(const std::string output_folder, const int time_interval,
                   const bool include_last_step)
      : periodic_task(time_interval, include_last_step),
        writer(specfem::io::wavefield_writer<IOLibrary<specfem::io::write> >(
            output_folder)) {}

  /**
   * @brief Check for keyboard interrupt and more, when running from Python
   *
   */
  void run(specfem::compute::assembly &assembly, const int istep) override {
    std::cout << "Writing wavefield files:" << std::endl;
    std::cout << "-------------------------------" << std::endl;
    writer.write(assembly, istep);
  }

  /**
   * @brief Write coordinates of wavefield data to disk.
   */
  void initialize(specfem::compute::assembly &assembly) override {
    std::cout << "Writing coordinate files:" << std::endl;
    std::cout << "-------------------------------" << std::endl;
    writer.write(assembly);
  }
};

} // namespace periodic_tasks
} // namespace specfem
