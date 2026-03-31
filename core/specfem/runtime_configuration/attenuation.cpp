#include "attenuation.hpp"
#include "specfem/units.hpp"
#include "yaml-cpp/yaml.h"
#include <ostream>

specfem::runtime_configuration::Attenuation::Attenuation(
    const YAML::Node &attenuation_node)
    : attenuation_enabled(false),
      reference_frequency(specfem::units::Hertz(-1.0)),
      attenuation_frequency_band(specfem::units::Hertz(-1.0),
                                 specfem::units::Hertz(-1.0)) {

  try {
    this->attenuation_enabled = attenuation_node["enabled"].as<bool>();

  } catch (YAML::ParserException &e) {
    std::ostringstream message;
    message << "Error reading attenuation configuration value: enabled. \n"
            << e.what();
    throw std::runtime_error(message.str());
  }

  try {
    this->reference_frequency =
        specfem::units::quantity_cast<specfem::units::Hertz>(
            attenuation_node["reference-frequency"].as<std::string>());

  } catch (YAML::ParserException &e) {
    std::ostringstream message;
    message << "Error reading attenuation configuration value: "
               "reference-frequency. \n"
            << e.what();
    throw std::runtime_error(message.str());
  }

  try {
    const YAML::Node band = attenuation_node["attenuation-frequency-band"];

    if (!band.IsSequence() || band.size() != 2) {
      throw std::runtime_error(
          "attenuation-frequency-band must be a sequence of exactly 2 "
          "values, e.g.:\n  attenuation-frequency-band:\n    - 0.1 Hz\n    - "
          "10.0 Hz");
    }
    specfem::units::Hertz minimum_frequency =
        specfem::units::quantity_cast<specfem::units::Hertz>(
            band[0].as<std::string>());
    specfem::units::Hertz maximum_frequency =
        specfem::units::quantity_cast<specfem::units::Hertz>(
            band[1].as<std::string>());

    this->attenuation_frequency_band =
        specfem::utilities::Band<specfem::units::Hertz>(minimum_frequency,
                                                        maximum_frequency);

  } catch (YAML::ParserException &e) {

    std::ostringstream message;
    message << "Error reading attenuation configuration value: "
               "attenuation-frequency-band. \n"
            << e.what();
    throw std::runtime_error(message.str());
  }
};
