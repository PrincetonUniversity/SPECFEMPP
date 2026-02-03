#include "external.hpp"
#include "enumerations/specfem_enums.hpp"
#include "io/seismogram/reader.hpp"
#include "kokkos_abstractions.h"
#include "specfem/utilities.hpp"
#include <fstream>
#include <tuple>
#include <vector>

specfem::source_time_functions::external::external(const YAML::Node &external,
                                                   const int nsteps,
                                                   const type_real dt)
    : nsteps_(nsteps), dt_(dt) {

  if (specfem::utilities::is_ascii_string(
          external["format"].as<std::string>()) ||
      !external["format"]) {
    this->format_ = specfem::enums::seismogram::format::ascii;
  } else {
    throw std::runtime_error("Only ASCII format is supported");
  }

  // Get the components from the file
  // Atleast one component is required
  if (const YAML::Node &stf = external["stf"]) {
    if (stf["X-component"] || stf["Z-component"]) {
      this->x_component_ =
          (stf["X-component"]) ? stf["X-component"].as<std::string>() : "";
      this->z_component_ =
          (stf["Z-component"]) ? stf["Z-component"].as<std::string>() : "";
      this->ncomponents_ = 2;
    } else if (stf["Y-component"]) {
      this->y_component_ = stf["Y-component"].as<std::string>();
      this->ncomponents_ = 1;
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
    if (this->ncomponents_ == 2) {
      if (this->x_component_.empty()) {
        return this->z_component_;
      } else {
        return this->x_component_;
      }
    } else {
      return this->y_component_;
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
  this->t0_ = time;

  std::getline(file, line);
  std::istringstream iss2(line);
  type_real time2, value2;
  iss2 >> time2 >> value2;
  this->dt_ = time2 - time;
  file.close();

  return;
}

void specfem::source_time_functions::external::compute_source_time_function(
    const type_real t0, const type_real dt, const int nsteps,
    Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>
        source_time_function) {

  const int ncomponents = source_time_function.extent(1);

  if (ncomponents != 2) {
    throw std::runtime_error("External source time function only supports 2 "
                             "components");
  }

  if (std::abs(t0 - this->t0_) > 1e-6) {
    throw std::runtime_error(
        "The start time of the external source time "
        "function does not match the simulation start time");
  }

  if (std::abs(dt - this->dt_) > 1e-6) {
    throw std::runtime_error(
        "The time step of the external source time "
        "function does not match the simulation time step");
  }

  std::vector<std::string> filename =
      (ncomponents == 2)
          ? std::vector<std::string>{ this->x_component_, this->z_component_ }
          : std::vector<std::string>{ this->y_component_ };

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

    Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace> data(
        "external", nsteps, 2);
    specfem::io::seismogram_reader reader(
        filename[icomp], specfem::enums::seismogram::format::ascii, data);
    reader.read();
    for (int i = 0; i < nsteps; i++) {
      source_time_function(i, icomp) = data(i, 1);
    }
  }
  return;
}

bool specfem::source_time_functions::external::operator==(
    const specfem::source_time_functions::stf &other) const {
  // First check base class equality
  if (!specfem::source_time_functions::stf::operator==(other))
    return false;

  // Then check if the other object is a dGaussian
  auto other_external =
      dynamic_cast<const specfem::source_time_functions::external *>(&other);
  if (!other_external)
    return false;

  return (this->x_component_ == other_external->x_component_ &&
          this->y_component_ == other_external->y_component_ &&
          this->z_component_ == other_external->z_component_ &&
          this->t0_ == other_external->t0_ &&
          this->dt_ == other_external->dt_ &&
          this->format_ == other_external->format_ &&
          this->ncomponents_ == other_external->ncomponents_ &&
          this->nsteps_ == other_external->nsteps_);
};

bool specfem::source_time_functions::external::operator!=(
    const specfem::source_time_functions::stf &other) const {
  return !(*this == other);
}

void specfem::source_time_functions::external::update_tshift(type_real tshift) {
  if (std::abs(tshift) > 1e-6) {
    throw std::runtime_error("Error: external source time function does not "
                             "support time shift");
  }
}

std::string specfem::source_time_functions::external::print() const {
  std::stringstream ss;
  ss << "External source time function: "
     << "\n"
     << "  X-component: " << this->x_component_ << "\n"
     << "  Y-component: " << this->y_component_ << "\n"
     << "  Z-component: " << this->z_component_ << "\n";
  return ss.str();
}
