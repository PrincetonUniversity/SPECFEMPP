// A well-formed plugin built with the C++ SDK layer. The configuration path
// "fail-create" makes its constructor throw, so create returns null with an
// error message.

#include "specfem/plugin/injection_plugin.hpp"
#include <stdexcept>
#include <string>
#include <vector>

namespace specfem::injection_loader_test {

class GoodPlugin : public specfem::plugin::InjectionPlugin {
public:
  GoodPlugin(
      const std::string &config_path,
      [[maybe_unused]] const specfem::plugin::SimulationInfo &info,
      [[maybe_unused]] const specfem::plugin::Communicator &communicator) {
    if (config_path == "fail-create")
      throw std::runtime_error("cannot read configuration 'fail-create'");
  }

  bool supports([[maybe_unused]] const specfem::plugin::Medium medium,
                [[maybe_unused]] const specfem::plugin::Property property,
                [[maybe_unused]] const specfem::plugin::Quantity quantity)
      const override {
    return false;
  }

  void setup([[maybe_unused]] const specfem::plugin::Points &points) override {}

  void fill([[maybe_unused]] const int first_step,
            [[maybe_unused]] const std::vector<specfem::plugin::Buffer>
                &buffers) override {}
};

} // namespace specfem::injection_loader_test

SPECFEM_DEFINE_INJECTION_PLUGIN("loader-good", "2.0.1",
                                specfem::injection_loader_test::GoodPlugin)
