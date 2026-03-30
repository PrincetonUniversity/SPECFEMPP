#include "attenuation.hpp"

#include "yaml-cpp/yaml.h"
#include <ostream>

specfem::runtime_configuration::Attenuation::Attenuation(
    const YAML::Node &attenuation_node) {
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
        attenuation_node["reference-frequency"].as<type_real>();

  } catch (YAML::ParserException &e) {
    std::ostringstream message;
    message << "Error reading attenuation configuration value: "
               "reference-frequency. \n"
            << e.what();
    throw std::runtime_error(message.str());
  }

  try {
    this->maximum_attenuation_frequency =
        attenuation_node["maximum-attenuation-frequency"].as<type_real>();

  } catch (YAML::ParserException &e) {

    std::ostringstream message;

    message << "Error reading attenuation configuration value: "
               "maximum-attenuation-frequency. \n"
            << e.what();
    throw std::runtime_error(message.str());
  }

  try {
    this->minimum_attenuation_frequency =
        attenuation_node["minimum-attenuation-frequency"].as<type_real>();

  } catch (YAML::ParserException &e) {

    std::ostringstream message;

    message << "Error reading attenuation configuration value: "
               "minimum-attenuation-frequency. \n"
            << e.what();
    throw std::runtime_error(message.str());
  }
};
