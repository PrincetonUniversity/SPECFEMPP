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

  std::vector<std::string> extentions = { ".BXX.semd", ".BXZ.semd" };

  // get t0 from file
  for (int icomp = 0; icomp < 2; ++icomp) {
    int file_components = 0;
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

      std::getline(file, line);
      std::istringstream iss2(line);
      type_real time2, value2;
      iss2 >> time2 >> value2;
      this->__dt = time2 - time;
      file.close();

      file_components++;
    }
    if (file_components != 1) {
      throw std::runtime_error(
          "External source time function file not found :" + this->__filename +
          extentions[icomp]);
    }
  }

  return;
}

void specfem::forcing_function::external::compute_source_time_function(
    const type_real t0, const type_real dt, const int nsteps,
    specfem::kokkos::HostView2d<type_real> source_time_function) {

  const int ncomponents = source_time_function.extent(1);

  if (ncomponents != 2) {
    throw std::runtime_error("External source time function only supports 2 "
                             "components");
  }

  if (std::abs(t0 - this->__t0) > 1e-6) {
    throw std::runtime_error(
        "The start time of the external source time "
        "function does not match the simulation start time");
  }

  if (std::abs(dt - this->__dt) > 1e-6) {
    throw std::runtime_error(
        "The time step of the external source time "
        "function does not match the simulation time step");
  }

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
