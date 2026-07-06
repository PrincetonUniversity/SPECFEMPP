#pragma once
#include "periodic_task.hpp"
#include "specfem/enums.hpp"
#include "specfem/io.hpp"
#include "specfem/logger.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace periodic_tasks {

/**
 * @brief Non-template base for wavefield_reader, allowing dynamic_cast from
 *        the solver without knowing the I/O backend at compile time.
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 */
template <specfem::element::dimension_tag DimensionTag>
class wavefield_reader_base : public periodic_task<DimensionTag> {
public:
  wavefield_reader_base(int time_interval, bool include_last_step)
      : periodic_task<DimensionTag>(time_interval, include_last_step) {}

  /**
   * @brief Read wavefield checkpoint including attenuation memory variables.
   *
   * Implemented by the concrete wavefield_reader<D, IOLibrary> subclass.
   */
  virtual void
  run_with_attenuation(specfem::assembly::assembly<DimensionTag> &assembly,
                       const int istep) = 0;
};

/**
 * @brief Periodic task to read wavefield data during simulation
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 * @tparam IOLibrary Template for the I/O library to use for reading
 */
template <specfem::element::dimension_tag DimensionTag,
          template <typename OpType> class IOLibrary>
class wavefield_reader : public wavefield_reader_base<DimensionTag> {
private:
  specfem::io::wavefield_reader<IOLibrary<specfem::io::read>> reader;

public:
  wavefield_reader(const std::string &output_folder, const int time_interval,
                   const bool include_last_step)
      : wavefield_reader_base<DimensionTag>(time_interval, include_last_step),
        reader(specfem::io::wavefield_reader<IOLibrary<specfem::io::read>>(
            output_folder)) {}

  /**
   * @brief Read wavefield data from file
   *
   */
  void run(specfem::assembly::assembly<DimensionTag> &assembly,
           const int istep) override {
    specfem::Logger::info("Reading wavefield files:");
    specfem::Logger::info("------------------------");
    reader.run(assembly, istep);
  }

  /**
   * @brief Read wavefield checkpoint including attenuation memory variables
   *
   * Called by the combined_undoatt solver at each subset boundary.
   */
  void run_with_attenuation(specfem::assembly::assembly<DimensionTag> &assembly,
                            const int istep) override {
    specfem::Logger::info("Reading wavefield checkpoint (with attenuation):");
    reader.run_with_attenuation(assembly, istep);
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
