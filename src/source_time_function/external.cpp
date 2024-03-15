#include "source_time_function/external.hpp"
#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include "reader/seismogram.hpp"
#include <tuple>
#include <vector>

specfem::forcing_function::external::external(const YAML::Node &external,
                                              const int nsteps,
                                              const type_real dt)
    : filename(external["filename"].as<std::string>()), nsteps(nsteps), dt(dt) {

  const specfem::enums::seismogram::format type = [&]() {
    if ((external["format"].as<std::string>() == "ascii") ||
        (external["format"].as<std::string>() == "ASCII")) {
      return specfem::enums::seismogram::format::ascii;
    } else {
      throw std::runtime_error("Only ASCII format is supported");
    }
  }();

  specfem::kokkos::HostView2d<type_real> data("external", nsteps, 2);
  specfem::reader::seismogram reader(filename, type, data);
  reader.read();

  this->t0 = data(0, 0);
  this->source_time_function = specfem::kokkos::HostView1d<type_real>(
      "specfem::forcing_function::external::source_time_function", nsteps);

  for (int i = 0; i < nsteps; ++i) {
    if (std::abs(data(i, 0) - this->t0 - i * this->dt) > 1e-6) {
      throw std::runtime_error("Error in external source time function data");
    }

    this->source_time_function(i) = data(i, 1);
  }

  return;
}
