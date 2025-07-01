#include "parameter_parser/receivers.hpp"
#include "utilities/strings.hpp"
#include "yaml-cpp/yaml.h"

// External Includes
#include <boost/tokenizer.hpp>
#include <fstream>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

std::vector<specfem::wavefield::type>
specfem::runtime_configuration::receivers::get_seismogram_types() const {

  std::vector<specfem::wavefield::type> stypes;

  // Allocate seismogram types
  assert(this->receivers_node["seismogram-type"].IsSequence());

  for (YAML::Node seismogram_type : this->receivers_node["seismogram-type"]) {
    if (specfem::utilities::is_displacement_string(
            seismogram_type.as<std::string>())) {
      stypes.push_back(specfem::wavefield::type::displacement);
    } else if (specfem::utilities::is_velocity_string(
                   seismogram_type.as<std::string>())) {
      stypes.push_back(specfem::wavefield::type::velocity);
    } else if (specfem::utilities::is_acceleration_string(
                   seismogram_type.as<std::string>())) {
      stypes.push_back(specfem::wavefield::type::acceleration);
    } else if (specfem::utilities::is_pressure_string(
                   seismogram_type.as<std::string>())) {
      stypes.push_back(specfem::wavefield::type::pressure);
    } else if (specfem::utilities::is_rotation_string(
                   seismogram_type.as<std::string>())) {
      stypes.push_back(specfem::wavefield::type::rotation);
    } else if (specfem::utilities::is_intrinsic_rotation_string(
                   seismogram_type.as<std::string>())) {
      stypes.push_back(specfem::wavefield::type::intrinsic_rotation);
    } else if (specfem::utilities::is_curl_string(
                   seismogram_type.as<std::string>())) {
      stypes.push_back(specfem::wavefield::type::curl);
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
