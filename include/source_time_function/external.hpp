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
      : __nsteps(nsteps), __dt(dt), __filename(filename) {}

  external(const YAML::Node &external, const int nsteps, const type_real dt);

  void compute_source_time_function(
      const type_real t0, const type_real dt, const int nsteps,
      specfem::kokkos::HostView1d<type_real> source_time_function) override {
    if (std::abs(t0 - this->__t0) > 1e-6 || std::abs(dt - this->__dt) > 1e-6) {
      throw std::runtime_error(
          "Error: Error reading source time function from file. "
          "Time step or time origin do not simulation values");
    }

    if (nsteps != this->__nsteps) {
      throw std::runtime_error(
          "Error: Error reading source time function from file. "
          "Number of time steps do not match simulation value");
    }
    source_time_function = this->__source_time_function;
  }

  void update_tshift(type_real tshift) override {
    if (std::abs(tshift) > 1e-6) {
      throw std::runtime_error("Error: external source time function does not "
                               "support time shift");
    }
  }

  std::string print() const override {
    std::stringstream ss;
    ss << "External source time function: " << this->__filename;
    return ss.str();
  }

private:
  int __nsteps;
  type_real __t0;
  type_real __dt;
  std::string __filename;
  specfem::kokkos::HostView1d<type_real> __source_time_function;
};
} // namespace forcing_function
} // namespace specfem

#endif /* SPECFEM_FORCING_FUNCTION_EXTERNAL_HPP */
