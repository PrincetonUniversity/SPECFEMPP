// Test plugin built with the C++ SDK layer. Values follow a simple formula so
// the tests can check exactly what was written, and the configuration path
// selects deliberate failures.

#include "specfem/plugin/injection_plugin.hpp"
#include <cstdint>
#include <new>
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
    if (config_path_ == "bad-alloc-in-create")
      throw std::bad_alloc();
  }

  // Elastic points (any parameterization except Cosserat) take every vector
  // quantity; acoustic points take displacement and potential rate.
  bool supports(const specfem::plugin::Medium medium,
                const specfem::plugin::Property property,
                const specfem::plugin::Quantity quantity) const override {
    using specfem::plugin::Medium;
    using specfem::plugin::Property;
    using specfem::plugin::Quantity;
    if (medium == Medium::elastic)
      return property != Property::isotropic_cosserat &&
             quantity != Quantity::potential_rate;
    if (medium == Medium::acoustic)
      return quantity == Quantity::displacement ||
             quantity == Quantity::potential_rate;
    return false;
  }

  // Records x + kappa for every point of every group; kappa is a property of
  // both elastic and acoustic isotropic points.
  void setup(const specfem::plugin::Points &points) override {
    if (config_path_ == "throw-in-setup")
      throw std::runtime_error("setup failed on purpose");
    if (config_path_ == "unsupported-in-setup")
      throw specfem::plugin::Unsupported("setup cannot serve this run");
    base_.clear();
    for (const auto &group : points.groups()) {
      std::vector<double> base;
      for (std::int64_t p = 0; p < group.size(); ++p)
        base.push_back(points.x(group.begin() + p) +
                       group.property(p, "kappa"));
      base_.push_back(std::move(base));
    }
  }

  // value(group, point, component, step)
  //   = 1000 * step + 10 * component + x(point) + kappa(point)
  void fill(const int first_step,
            const std::vector<specfem::plugin::Buffer> &buffers) override {
    if (config_path_ == "throw-in-fill")
      throw std::runtime_error("fill failed on purpose");
    if (config_path_ == "bad-alloc-in-fill")
      throw std::bad_alloc();
    if (config_path_ == "long-error-in-fill")
      throw std::runtime_error(std::string(2000, 'e'));
    for (std::size_t g = 0; g < buffers.size(); ++g) {
      const auto &buffer = buffers[g];
      if (config_path_ == "write-out-of-range")
        buffer.set(buffer.number_of_points(), 0, 0, 0.0);
      for (std::int64_t p = 0; p < buffer.number_of_points(); ++p)
        for (int c = 0; c < buffer.number_of_components(); ++c)
          for (int s = 0; s < buffer.number_of_steps(); ++s)
            buffer.set(p, c, s,
                       1000.0 * (first_step + s) + 10.0 * c + base_[g][p]);
    }
  }

private:
  std::string config_path_;
  std::vector<std::vector<double>> base_;
};

} // namespace specfem::plugin_sdk_test

SPECFEM_DEFINE_INJECTION_PLUGIN("test-good", "1.2.3",
                                specfem::plugin_sdk_test::GoodPlugin)
