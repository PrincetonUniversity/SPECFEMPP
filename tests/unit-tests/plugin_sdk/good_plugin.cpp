// Test plugin built with the C++ SDK layer. Values follow a simple formula so
// the tests can check exactly what was written, and the configuration path
// selects deliberate failures.

#include "specfem/plugin/injection_plugin.hpp"
#include <stdexcept>
#include <string>
#include <vector>

namespace specfem::plugin_sdk_test {

class GoodPlugin : public specfem::plugin::InjectionPlugin {
public:
  GoodPlugin(const std::string &config_path,
             [[maybe_unused]] const specfem::plugin::SimulationInfo &info,
             [[maybe_unused]] const specfem::plugin::Communicator &communicator)
      : config_path_(config_path) {
    if (config_path_ == "throw-in-create")
      throw std::runtime_error("create failed on purpose");
  }

  bool supports(const specfem::plugin::Medium medium,
                const specfem::plugin::Quantity quantity) const override {
    return medium == specfem::plugin::Medium::elastic &&
           quantity != specfem::plugin::Quantity::potential_rate;
  }

  void setup(const specfem::plugin::Points &points,
             const specfem::plugin::Request &request) override {
    if (config_path_ == "throw-in-setup")
      throw std::runtime_error("setup failed on purpose");
    x_.clear();
    for (std::int64_t i = 0; i < points.size(); ++i)
      x_.push_back(points.x(i));
    request_ = request;
  }

  // value(point, component, step) = 1000 * step + 10 * component + x(point)
  void fill(const int first_step, specfem::plugin::Buffer &buffer) override {
    if (config_path_ == "throw-in-fill")
      throw std::runtime_error("fill failed on purpose");
    const int ncomponents =
        request_.component_count(specfem::plugin::Medium::elastic);
    for (std::size_t point = 0; point < x_.size(); ++point)
      for (int component = 0; component < ncomponents; ++component)
        for (int step = 0; step < buffer.number_of_steps(); ++step)
          buffer.set(static_cast<std::int64_t>(point), component, step,
                     1000.0 * (first_step + step) + 10.0 * component +
                         x_[point]);
  }

private:
  std::string config_path_;
  std::vector<double> x_;
  specfem::plugin::Request request_;
};

} // namespace specfem::plugin_sdk_test

SPECFEMPP_DEFINE_INJECTION_PLUGIN("test-good", "1.2.3",
                                  specfem::plugin_sdk_test::GoodPlugin)
