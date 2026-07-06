#pragma once
#include "periodic_task.hpp"
#include "specfem/enums.hpp"
#include "specfem/io.hpp"
#include "specfem/logger.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace periodic_tasks {

/**
 * @brief Periodic task to write wavefield data during simulation
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 * @tparam IOLibrary Template for the I/O library to use for writing
 */
template <specfem::element::dimension_tag DimensionTag,
          template <typename OpType> class IOLibrary>
class wavefield_writer : public periodic_task<DimensionTag> {
private:
  specfem::io::wavefield_writer<IOLibrary<specfem::io::write>> writer;
  bool for_adjoint_simulations;

public:
  wavefield_writer(const std::string &output_folder, const int time_interval,
                   const bool include_last_step,
                   const bool for_adjoint_simulations)
      : periodic_task<DimensionTag>(time_interval, include_last_step),
        writer(specfem::io::wavefield_writer<IOLibrary<specfem::io::write>>(
            output_folder, for_adjoint_simulations)),
        for_adjoint_simulations(for_adjoint_simulations) {}

  /**
   * @brief Write wavefield data to file
   *
   * When attenuation checkpointing is enabled the SLS memory variables are
   * written alongside the kinematic fields; this is handled internally by the
   * writer based on its for_adjoint_simulations flag.
   */
  void run(specfem::assembly::assembly<DimensionTag> &assembly,
           const int istep) override {
    if (for_adjoint_simulations) {
      specfem::Logger::info("Writing wavefield checkpoint (with attenuation):");
    } else {
      std::cout << "Writing wavefield files:" << std::endl;
      std::cout << "-------------------------------" << std::endl;
    }
    writer.run(assembly, istep);
  }

  /**
   * @brief Write coordinates of wavefield data to disk.
   */
  void
  initialize(specfem::assembly::assembly<DimensionTag> &assembly) override {
    specfem::Logger::info("Writing coordinate files:");
    specfem::Logger::info("-------------------------");
    writer.initialize(assembly);
    if (for_adjoint_simulations) {
      run(assembly, 0);
    }
  }

  void finalize(specfem::assembly::assembly<DimensionTag> &assembly) override {
    specfem::Logger::info("Finalizing wavefield files:");
    specfem::Logger::info("---------------------------");
    writer.finalize(assembly);
  }

  specfem::periodic_tasks::type get_type() const override {
    return specfem::periodic_tasks::type::wavefield_writer;
  }
};

} // namespace periodic_tasks
} // namespace specfem
