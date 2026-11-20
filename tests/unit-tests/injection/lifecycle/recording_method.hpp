#pragma once

#include "specfem/injection/method/injection_method.hpp"
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

namespace specfem::injection_lifecycle_test {

/**
 * @brief Calls observed by RecordingMethod since the last reset.
 */
struct MethodRecord {
  int setups = 0;
  int window_steps = 0;             ///< Window size of the session handed over
  std::vector<int> prepared_steps;  ///< istep of every prepare_step
  std::vector<int> elastic_applies; ///< istep of every elastic apply
};

/** @brief Calls observed so far. */
inline MethodRecord &method_record() {
  static MethodRecord record;
  return record;
}

/**
 * @brief A no-op injection method that requests elastic/isotropic velocity and
 * records the lifecycle calls it receives.
 */
class RecordingMethod final : public specfem::injection::InjectionMethod<
                                  specfem::element::dimension_tag::dim3> {
public:
  std::string_view name() const override { return "recording"; }

  specfem::injection::QuantityPolicy quantities() const override {
    return
        [](const specfem::element::medium_tag medium,
           const specfem::element::property_tag property)
            -> std::optional<std::vector<specfem::injection::QuantityRequest>> {
          if (medium == specfem::element::medium_tag::elastic &&
              property == specfem::element::property_tag::isotropic)
            return std::vector<specfem::injection::QuantityRequest>{
              { specfem_quantity_velocity, 3 }
            };
          return std::nullopt;
        };
  }

  void
  setup(const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
            &,
        specfem::injection::ProviderSession session) override {
    ++method_record().setups;
    method_record().window_steps = session.window_steps();
    session_.emplace(std::move(session));
  }

  void prepare_step(const int istep) override {
    method_record().prepared_steps.push_back(istep);
    session_->ensure_window(istep);
  }

  void
  apply(specfem::assembly::assembly<specfem::element::dimension_tag::dim3> &,
        const specfem::element::medium_tag medium,
        specfem::simulation::field_type, const int istep) override {
    if (medium == specfem::element::medium_tag::elastic)
      method_record().elastic_applies.push_back(istep);
  }

private:
  std::optional<specfem::injection::ProviderSession> session_;
};

} // namespace specfem::injection_lifecycle_test
