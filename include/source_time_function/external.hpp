#ifndef SPECFEM_FORCING_FUNCTION_EXTERNAL_HPP
#define SPECFEM_FORCING_FUNCTION_EXTERNAL_HPP

#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include "source_time_function/source_time_function.hpp"
#include "yaml-cpp/yaml.h"
#include <tuple>
#include <vector>

namespace specfem {
namespace forcing_function {
class external : public stf {
public:
  external(const YAML::Node &external, const int nsteps, const type_real dt);

  void compute_source_time_function(
      const type_real t0, const type_real dt, const int nsteps,
      specfem::kokkos::HostView2d<type_real> source_time_function) override;

  void update_tshift(type_real tshift) override {
    if (std::abs(tshift) > 1e-6) {
      throw std::runtime_error("Error: external source time function does not "
                               "support time shift");
    }
  }

  std::string print() const override {
    std::stringstream ss;
    ss << "External source time function: "
       << "\n"
       << "  X-component: " << this->x_component << "\n"
       << "  Y-component: " << this->y_component << "\n"
       << "  Z-component: " << this->z_component << "\n";
    return ss.str();
  }

  type_real get_t0() const override { return this->__t0; }

  type_real get_tshift() const override { return 0.0; }

private:
  int __nsteps;
  type_real __t0;
  type_real __dt;
  specfem::enums::seismogram::format type;
  int ncomponents;
  std::string x_component = "";
  std::string y_component = "";
  std::string z_component = "";
};
} // namespace forcing_function
} // namespace specfem

#endif /* SPECFEM_FORCING_FUNCTION_EXTERNAL_HPP */
