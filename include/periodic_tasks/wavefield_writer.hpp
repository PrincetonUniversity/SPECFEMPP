#pragma once
#include "enumerations/interface.hpp"
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
  wavefield_writer(const std::string &output_folder, const int time_interval,
                   const bool include_last_step,
                   const bool save_boundary_values)
      : periodic_task(time_interval, include_last_step),
        writer(specfem::io::wavefield_writer<IOLibrary<specfem::io::write> >(
            output_folder, save_boundary_values)) {}

  /**
   * @brief Check for keyboard interrupt and more, when running from Python
   *
   */
  void
  run(specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly,
      const int istep) override {
    std::cout << "Writing wavefield files:" << std::endl;
    std::cout << "-------------------------------" << std::endl;
    writer.run(assembly, istep);
  }

  /**
   * @brief Write coordinates of wavefield data to disk.
   */
  void initialize(specfem::assembly::assembly<specfem::dimension::type::dim2>
                      &assembly) override {
    std::cout << "Writing coordinate files:" << std::endl;
    std::cout << "-------------------------------" << std::endl;
    writer.initialize(assembly);
  }

  void finalize(specfem::assembly::assembly<specfem::dimension::type::dim2>
                    &assembly) override {
    std::cout << "Finalizing wavefield files:" << std::endl;
    std::cout << "-------------------------------" << std::endl;
    writer.finalize(assembly);
  }
};

} // namespace periodic_tasks
} // namespace specfem
