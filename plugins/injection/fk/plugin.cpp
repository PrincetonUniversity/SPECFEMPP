// Entry point of the FK injection plugin module. Skeleton: serves nothing yet.

#include "specfem/fk_plugin.hpp"
#include "specfem/fk_plugin/impl/kokkos_lifetime.hpp"
#include "specfem/plugin/injection_plugin.hpp"

#include <string>
#include <vector>

namespace specfem {
namespace fk_plugin {

/**
 * @brief FK incident-field provider (skeleton).
 *
 * Supports nothing; setup and fill report the run as unsupported.
 */
class FkInjectionPlugin : public specfem::plugin::InjectionPlugin {
public:
  /**
   * @brief Start the plugin's private Kokkos.
   * @param config_path Path of the plugin configuration file (unused)
   * @param info Simulation parameters (unused)
   * @param communicator MPI communicator description (unused)
   */
  FkInjectionPlugin(const std::string &config_path,
                    const specfem::plugin::SimulationInfo &info,
                    const specfem::plugin::Communicator &communicator) {}

  bool supports(specfem::plugin::Medium medium,
                specfem::plugin::Property property,
                specfem::plugin::Quantity quantity) const override {
    return false;
  }

  void setup(const specfem::plugin::Points &points) override {
    throw specfem::plugin::Unsupported(
        "FK plugin skeleton: physics not implemented yet");
  }

  void fill(int first_step,
            const std::vector<specfem::plugin::Buffer> &buffers) override {
    throw specfem::plugin::Unsupported(
        "FK plugin skeleton: physics not implemented yet");
  }

private:
  // First member: outlives every Kokkos object of the plugin.
  specfem::fk_plugin_impl::KokkosLifetime kokkos_lifetime_;
};

} // namespace fk_plugin
} // namespace specfem

SPECFEM_DEFINE_INJECTION_PLUGIN("fk", "0.0.1",
                                specfem::fk_plugin::FkInjectionPlugin)
