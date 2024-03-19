#include "source_time_function/external.hpp"
#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include "reader/seismogram.hpp"
#include <tuple>
#include <vector>

specfem::forcing_function::external::external(const YAML::Node &external,
                                              const int nsteps,
                                              const type_real dt)
    : __nsteps(nsteps), __dt(dt),
      __filename(external["filename"].as<std::string>()) {

  const specfem::enums::seismogram::format type = [&]() {
    if ((external["format"].as<std::string>() == "ascii") ||
        (external["format"].as<std::string>() == "ASCII")) {
      return specfem::enums::seismogram::format::ascii;
    } else {
      throw std::runtime_error("Only ASCII format is supported");
    }
  }();

  specfem::kokkos::HostView2d<type_real> data("external", nsteps, 2);
  specfem::reader::seismogram reader(this->__filename, type, data);
  reader.read();

  this->__t0 = data(0, 0);
  this->__source_time_function = specfem::kokkos::HostView1d<type_real>(
      "specfem::forcing_function::external::source_time_function", nsteps);

  for (int i = 0; i < nsteps; ++i) {
    this->__source_time_function(i) = data(i, 1);
  }

  return;
}
