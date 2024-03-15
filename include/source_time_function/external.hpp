#ifndef SPECFEM_FORCING_FUNCTION_EXTERNAL_HPP
#define SPECFEM_FORCING_FUNCTION_EXTERNAL_HPP

#include "kokkos_abstractions.h"
#include "source_time_function/source_time_function.hpp"
#include "yaml-cpp/yaml.h"
#include <tuple>
#include <vector>

namespace specfem {
namespace forcing_function {
class external : public stf {
public:
  external(const std::string &filename, const int nsteps, const type_real dt)
      : filename(filename), nsteps(nsteps), dt(dt) {}

  external(const YAML::Node &external, const int nsteps, const type_real dt);

  void compute_source_time_function(
      specfem::kokkos::HostView1d<type_real> source_time_function) override {
    source_time_function = this->source_time_function;
  }

  void update_tshift(type_real tshift) override {
    if (std::abs(tshift) > 1e-6) {
      throw std::runtime_error("Error: external source time function does not "
                               "support time shift");
    }
  }

  std::string print() const override {
    std::stringstream ss;
    ss << "External source time function: " << this->filename;
    return ss.str();
  }

private:
  int nsteps;
  type_real t0;
  type_real dt;
  std::string filename;
  specfem::kokkos::HostView1d<type_real> source_time_function;
};
} // namespace forcing_function
} // namespace specfem

#endif /* SPECFEM_FORCING_FUNCTION_EXTERNAL_HPP */
