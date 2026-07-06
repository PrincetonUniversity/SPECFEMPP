#pragma once
#include "periodic_task.hpp"
#include "specfem/enums.hpp"
#include "specfem/io.hpp"
#include "specfem/logger.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace periodic_tasks {

/**
 * @brief Periodic task to read wavefield data during simulation
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 * @tparam IOLibrary Template for the I/O library to use for reading
 */
template <specfem::element::dimension_tag DimensionTag,
          template <typename OpType> class IOLibrary>
class wavefield_reader : public periodic_task<DimensionTag> {
private:
  specfem::io::wavefield_reader<IOLibrary<specfem::io::read>> reader;

public:
  wavefield_reader(const std::string &output_folder, const int time_interval,
                   const bool include_last_step)
      : periodic_task<DimensionTag>(time_interval, include_last_step),
        reader(specfem::io::wavefield_reader<IOLibrary<specfem::io::read>>(
            output_folder)) {}

  /**
   * @brief Read wavefield data from file
   *
   * When attenuation checkpointing is enabled the SLS memory variables are
   * loaded alongside the kinematic fields; this is handled internally by the
   * reader.
   */
  void run(specfem::assembly::assembly<DimensionTag> &assembly,
           const int istep) override {
    specfem::Logger::info("Reading wavefield checkpoint:");
    reader.run(assembly, istep);
  }

  void
  initialize(specfem::assembly::assembly<DimensionTag> &assembly) override {
    specfem::Logger::info("Reading coordinate files:");
    specfem::Logger::info("-------------------------");
    reader.initialize(assembly);
  }
};

} // namespace periodic_tasks
} // namespace specfem
