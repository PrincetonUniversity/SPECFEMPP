// Minimal plugin used to check that the installed SDK is self-contained.

#include "specfem/plugin/injection_plugin.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace specfem::plugin_sdk_consumer {

class ZeroPlugin : public specfem::plugin::InjectionPlugin {
public:
  ZeroPlugin(const std::string &, const specfem::plugin::SimulationInfo &,
             const specfem::plugin::Communicator &) {}

  bool supports(specfem::plugin::Medium, specfem::plugin::Property,
                specfem::plugin::Quantity) const override {
    return true;
  }

  void setup(const specfem::plugin::Points &) override {}

  void fill(int, const std::vector<specfem::plugin::Buffer> &buffers) override {
    for (const auto &buffer : buffers)
      for (std::int64_t point = 0; point < buffer.number_of_points(); ++point)
        for (int component = 0; component < buffer.number_of_components();
             ++component)
          for (int step = 0; step < buffer.number_of_steps(); ++step)
            buffer.set(point, component, step, 0.0);
  }
};

} // namespace specfem::plugin_sdk_consumer

SPECFEM_DEFINE_INJECTION_PLUGIN("consumer", "0.0.1",
                                specfem::plugin_sdk_consumer::ZeroPlugin)
