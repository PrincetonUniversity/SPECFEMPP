#pragma once
#include "periodic_task.hpp"
#include "specfem/enums.hpp"
#include "specfem/io.hpp"
#include "specfem/logger.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace periodic_tasks {

/**
 * @brief Non-template base for wavefield_writer, allowing dynamic_cast from
 *        the solver without knowing the I/O backend at compile time.
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 */
template <specfem::element::dimension_tag DimensionTag>
class wavefield_writer_base : public periodic_task<DimensionTag> {
public:
  wavefield_writer_base(int time_interval, bool include_last_step)
      : periodic_task<DimensionTag>(time_interval, include_last_step) {}

  /**
   * @brief Write wavefield snapshot including attenuation memory variables.
   *
   * Implemented by the concrete wavefield_writer<D, IOLibrary> subclass.
   */
  virtual void
  run_with_attenuation(specfem::assembly::assembly<DimensionTag> &assembly,
                       const int istep) = 0;
};

/**
 * @brief Periodic task to write wavefield data during simulation
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 * @tparam IOLibrary Template for the I/O library to use for writing
 */
template <specfem::element::dimension_tag DimensionTag,
          template <typename OpType> class IOLibrary>
class wavefield_writer : public wavefield_writer_base<DimensionTag> {
private:
  specfem::io::wavefield_writer<IOLibrary<specfem::io::write>> writer;
  bool save_attenuation;

public:
  wavefield_writer(const std::string &output_folder, const int time_interval,
                   const bool include_last_step,
                   const bool save_boundary_values)
      : wavefield_writer_base<DimensionTag>(time_interval, include_last_step),
        writer(specfem::io::wavefield_writer<IOLibrary<specfem::io::write>>(
            output_folder, save_boundary_values)),
        save_attenuation(save_boundary_values) {}

  /**
   * @brief Write wavefield data to file
   *
   */
  void run(specfem::assembly::assembly<DimensionTag> &assembly,
           const int istep) override {
    if (save_attenuation) {
      run_with_attenuation(assembly, istep);
    } else {
      std::cout << "Writing wavefield files:" << std::endl;
      std::cout << "-------------------------------" << std::endl;
      writer.run(assembly, istep);
    }
  }

  /**
   * @brief Write wavefield snapshot including attenuation memory variables
   *
   * Called by forward simulations configured for adjoint checkpoints.
   */
  void run_with_attenuation(specfem::assembly::assembly<DimensionTag> &assembly,
                            const int istep) override {
    specfem::Logger::info("Writing wavefield checkpoint (with attenuation):");
    writer.run_with_attenuation(assembly, istep);
  }

  /**
   * @brief Write coordinates of wavefield data to disk.
   */
  void
  initialize(specfem::assembly::assembly<DimensionTag> &assembly) override {
    specfem::Logger::info("Writing coordinate files:");
    specfem::Logger::info("-------------------------");
    writer.initialize(assembly);
    if (save_attenuation) {
      run_with_attenuation(assembly, 0);
    }
  }

  void finalize(specfem::assembly::assembly<DimensionTag> &assembly) override {
    specfem::Logger::info("Finalizing wavefield files:");
    specfem::Logger::info("---------------------------");
    writer.finalize(assembly);
  }
};

} // namespace periodic_tasks
} // namespace specfem
