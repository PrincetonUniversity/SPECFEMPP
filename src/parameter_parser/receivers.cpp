#include "parameter_parser/receivers.hpp"
#include "yaml-cpp/yaml.h"

// External Includes
#include <boost/tokenizer.hpp>
#include <fstream>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

std::vector<specfem::enums::seismogram::type>
specfem::runtime_configuration::receivers::get_seismogram_types() const {

  std::vector<specfem::enums::seismogram::type> stypes;

  // Allocate seismogram types
  assert(this->receivers_node["seismogram-type"].IsSequence());

  for (YAML::Node seismogram_type : this->receivers_node["seismogram-type"]) {
    if (seismogram_type.as<std::string>() == "displacement") {
      stypes.push_back(specfem::enums::seismogram::type::displacement);
    } else if (seismogram_type.as<std::string>() == "velocity") {
      stypes.push_back(specfem::enums::seismogram::type::velocity);
    } else if (seismogram_type.as<std::string>() == "acceleration") {
      stypes.push_back(specfem::enums::seismogram::type::acceleration);
    } else if (seismogram_type.as<std::string>() == "pressure") {
      stypes.push_back(specfem::enums::seismogram::type::pressure);
    } else {
      std::ostringstream message;

      message << "Error reading specfem receiver configuration. \n";
      message << "Unknown seismogram type: "
              << seismogram_type.as<std::string>() << "\n";

      std::runtime_error(message.str());
    }
  }
  return stypes;
}
