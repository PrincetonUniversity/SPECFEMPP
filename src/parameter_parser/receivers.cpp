#include "parameter_parser/receivers.hpp"
#include "constants.hpp"
#include "yaml-cpp/yaml.h"
#include <ostream>
#include <string>

specfem::runtime_configuration::receivers::receivers(const YAML::Node &Node) {
  try {
    *this = specfem::runtime_configuration::receivers(
        Node["stations-file"].as<std::string>(), Node["angle"].as<type_real>(),
        Node["nstep_between_samples"].as<int>());
  } catch (YAML::ParserException &e) {
    std::ostringstream message;

    message << "Error reading specfem receiver configuration. \n" << e.what();

    throw std::runtime_error(message.str());
  }

  // Allocate seismogram types
  assert(Node["seismogram-type"].IsSequence());

  for (YAML::Node seismogram_type : Node["seismogram-type"]) {
    if (seismogram_type.as<std::string>() == "displacement") {
      this->stypes.push_back(specfem::enums::seismogram::type::displacement);
    } else if (seismogram_type.as<std::string>() == "velocity") {
      this->stypes.push_back(specfem::enums::seismogram::type::velocity);
    } else if (seismogram_type.as<std::string>() == "acceleration") {
      this->stypes.push_back(specfem::enums::seismogram::type::acceleration);
    } else {
      std::ostringstream message;

      message << "Error reading specfem receiver configuration. \n";
      message << "Unknown seismogram type: "
              << seismogram_type.as<std::string>() << "\n";

      std::runtime_error(message.str());
    }
  }
}
