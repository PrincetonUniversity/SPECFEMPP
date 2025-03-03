#include "source_time_function/external.hpp"
#include "IO/seismogram/reader.hpp"
#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include <fstream>
#include <tuple>
#include <vector>

specfem::forcing_function::external::external(const YAML::Node &external,
                                              const int nsteps,
                                              const type_real dt)
    : __nsteps(nsteps), __dt(dt) {

  if ((external["format"].as<std::string>() == "ascii") ||
      (external["format"].as<std::string>() == "ASCII") ||
      !external["format"]) {
    this->type = specfem::enums::seismogram::format::ascii;
  } else {
    throw std::runtime_error("Only ASCII format is supported");
  }

  // Get the components from the file
  // Atleast one component is required
  if (const YAML::Node &stf = external["stf"]) {
    if (stf["X-component"] || stf["Z-component"]) {
      this->x_component =
          (stf["X-component"]) ? stf["X-component"].as<std::string>() : "";
      this->z_component =
          (stf["Z-component"]) ? stf["Z-component"].as<std::string>() : "";
      this->ncomponents = 2;
    } else if (stf["Y-component"]) {
      this->y_component = stf["Y-component"].as<std::string>();
      this->ncomponents = 1;
    } else {
      throw std::runtime_error("Error: External source time function requires "
                               "at least one component");
    }
  } else {
    throw std::runtime_error("Error: External source time function requires "
                             "at least one component");
  }

  // Get t0 and dt from the file
  const std::string filename = [&]() -> std::string {
    if (this->ncomponents == 2) {
      if (this->x_component.empty()) {
        return this->z_component;
      } else {
        return this->x_component;
      }
    } else {
      return this->y_component;
    }
  }();

  std::ifstream file(filename);
  if (!file.good()) {
    throw std::runtime_error("Error: External source time function file " +
                             filename + " does not exist");
  }

  std::string line;
  std::getline(file, line);
  std::istringstream iss(line);
  type_real time, value;
  if (!(iss >> time >> value)) {
    throw std::runtime_error("Seismogram file " + filename +
                             " is not formatted correctly");
  }
  this->__t0 = time;

  std::getline(file, line);
  std::istringstream iss2(line);
  type_real time2, value2;
  iss2 >> time2 >> value2;
  this->__dt = time2 - time;
  file.close();

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

  std::vector<std::string> filename =
      (ncomponents == 2)
          ? std::vector<std::string>{ this->x_component, this->z_component }
          : std::vector<std::string>{ this->y_component };

  // Check if files exist
  for (int icomp = 0; icomp < ncomponents; ++icomp) {
    // Skip empty filenames
    if (filename[icomp].empty())
      continue;

    std::ifstream file(filename[icomp]);
    if (!file.good()) {
      throw std::runtime_error("Error: External source time function file " +
                               filename[icomp] + " does not exist");
    }
  }

  // set source time function to 0
  for (int i = 0; i < nsteps; i++) {
    for (int icomp = 0; icomp < ncomponents; ++icomp) {
      source_time_function(i, icomp) = 0.0;
    }
  }

  for (int icomp = 0; icomp < ncomponents; ++icomp) {
    if (filename[icomp].empty())
      continue;

    specfem::kokkos::HostView2d<type_real> data("external", nsteps, 2);
    specfem::IO::seismogram_reader reader(
        filename[icomp], specfem::enums::seismogram::format::ascii, data);
    reader.read();
    for (int i = 0; i < nsteps; i++) {
      source_time_function(i, icomp) = data(i, 1);
    }
  }
  return;
}
