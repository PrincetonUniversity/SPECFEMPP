#include "parameter_parser/quadrature.hpp"
#include "quadrature/interface.hpp"
#include "yaml-cpp/yaml.h"
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>

specfem::quadrature::quadratures
specfem::runtime_configuration::quadrature::instantiate() {

  specfem::quadrature::gll::gll gll(this->alpha, this->beta, this->ngll);

  return specfem::quadrature::quadratures(gll);
}

specfem::runtime_configuration::quadrature::quadrature(
    const std::string quadrature) {

  if (quadrature == "GLL4") {
    *this = specfem::runtime_configuration::quadrature(0.0, 0.0, 5);
  } else if (quadrature == "GLL7") {
    *this = specfem::runtime_configuration::quadrature(0.0, 0.0, 8);
  } else {
    std::ostringstream message;
    message << "Error reading quadrature argument. " << quadrature
            << " hasn't been implemented yet.\n";
    throw std::runtime_error(message.str());
  }

  return;
}

specfem::runtime_configuration::quadrature::quadrature(const YAML::Node &Node) {
  try {
    *this = specfem::runtime_configuration::quadrature(
        Node["alpha"].as<type_real>(), Node["beta"].as<type_real>(),
        Node["ngll"].as<int>());
  } catch (YAML::InvalidNode &e) {
    try {
      *this = specfem::runtime_configuration::quadrature(
          Node["quadrature-type"].as<std::string>());
    } catch (YAML::InvalidNode &e) {
      std::ostringstream message;

      message << "Error reading quarature config. \n" << e.what();

      throw std::runtime_error(message.str());
    }
  }

  return;
}
