#include "receivers.hpp"
#include "specfem/utilities.hpp"
#include "yaml-cpp/yaml.h"

// External Includes
#include <boost/tokenizer.hpp>
#include <fstream>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

std::vector<specfem::enums::wavefield>
specfem::runtime_configuration::receivers::get_seismogram_types() const {

  std::vector<specfem::enums::wavefield> stypes;

  // Allocate seismogram types
  assert(this->receivers_node["seismogram-type"].IsSequence());

  for (YAML::Node seismogram_type : this->receivers_node["seismogram-type"]) {
    if (specfem::utilities::is_displacement_string(
            seismogram_type.as<std::string>())) {
      stypes.push_back(specfem::enums::wavefield::displacement);
    } else if (specfem::utilities::is_velocity_string(
                   seismogram_type.as<std::string>())) {
      stypes.push_back(specfem::enums::wavefield::velocity);
    } else if (specfem::utilities::is_acceleration_string(
                   seismogram_type.as<std::string>())) {
      stypes.push_back(specfem::enums::wavefield::acceleration);
    } else if (specfem::utilities::is_pressure_string(
                   seismogram_type.as<std::string>())) {
      stypes.push_back(specfem::enums::wavefield::pressure);
    } else if (specfem::utilities::is_rotation_string(
                   seismogram_type.as<std::string>())) {
      stypes.push_back(specfem::enums::wavefield::rotation);
    } else if (specfem::utilities::is_intrinsic_rotation_string(
                   seismogram_type.as<std::string>())) {
      stypes.push_back(specfem::enums::wavefield::intrinsic_rotation);
    } else if (specfem::utilities::is_curl_string(
                   seismogram_type.as<std::string>())) {
      stypes.push_back(specfem::enums::wavefield::curl);
    } else {
      std::ostringstream message;

      message << "Error reading specfem receiver configuration. (" << __FILE__
              << ":" << __LINE__ << ")\n";
      message << "Unknown seismogram type: "
              << seismogram_type.as<std::string>() << "\n";

      throw std::runtime_error(message.str());
    }
  }
  return stypes;
}
