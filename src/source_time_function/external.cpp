#include "source_time_function/external.hpp"
#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include "reader/seismogram.hpp"
#include <fstream>
#include <tuple>
#include <vector>

specfem::forcing_function::external::external(const YAML::Node &external,
                                              const int nsteps,
                                              const type_real dt)
    : __nsteps(nsteps), __dt(dt),
      __filename(external["stf-file"].as<std::string>()) {

  if ((external["format"].as<std::string>() == "ascii") ||
      (external["format"].as<std::string>() == "ASCII")) {
    this->type = specfem::enums::seismogram::format::ascii;
  } else {
    throw std::runtime_error("Only ASCII format is supported");
  }

  std::vector<std::string> extentions = { ".BXX.semd", ".BXY.semd" };

  int file_components = 0;

  // get t0 from file
  for (int icomp = 0; icomp < 2; ++icomp) {
    std::ifstream file(this->__filename + extentions[icomp]);
    if (file.good()) {
      // read the first line
      std::string line;
      std::getline(file, line);
      std::istringstream iss(line);
      type_real time, value;
      if (!(iss >> time >> value)) {
        throw std::runtime_error("Seismogram file " + this->__filename +
                                 " is not formatted correctly");
      }
      this->__t0 = time;
      file_components++;
    }
    file.close();
  }

  if (file_components != 1) {
    throw std::runtime_error(
        "Wrong number for files found for external source: " +
        this->__filename);
  }

  return;
}

void specfem::forcing_function::external::compute_source_time_function(
    const type_real t0, const type_real dt, const int nsteps,
    specfem::kokkos::HostView2d<type_real> source_time_function) {

  const int ncomponents = source_time_function.extent(1);

  const auto extention = [&ncomponents]() -> std::vector<std::string> {
    if (ncomponents == 2) {
      return { ".BXX.semd", ".BXZ.semd" };
    } else if (ncomponents == 1) {
      return { ".BXY.semd" };
    } else {
      throw std::runtime_error("Invalid number of components");
    }
  }();

  // Check if files exist
  for (int icomp = 0; icomp < ncomponents; ++icomp) {
    std::ifstream file(this->__filename + extention[icomp]);
    if (!file.good()) {
      throw std::runtime_error("Error: External source time function file " +
                               this->__filename + extention[icomp] +
                               " does not exist");
    }
  }

  for (int icomp = 0; icomp < ncomponents; ++icomp) {
    specfem::kokkos::HostView2d<type_real> data("external", nsteps, 2);
    specfem::reader::seismogram reader(
        this->__filename + extention[icomp],
        specfem::enums::seismogram::format::ascii, data);
    reader.read();
    for (int i = 0; i < nsteps; i++) {
      source_time_function(i, icomp) = data(i, 1);
    }
  }
  return;
}
