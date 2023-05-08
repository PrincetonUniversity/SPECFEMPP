#include "parameter_parser/interface.hpp"
#include "quadrature/interface.hpp"
#include "yaml-cpp/yaml.h"
#include <ostream>
#include <string>
#include <tuple>

std::tuple<specfem::quadrature::quadrature *, specfem::quadrature::quadrature *>
specfem::runtime_configuration::quadrature::instantiate() {

  specfem::quadrature::quadrature *gllx =
      new specfem::quadrature::gll::gll(this->alpha, this->beta, this->ngllx);
  specfem::quadrature::quadrature *gllz =
      new specfem::quadrature::gll::gll(this->alpha, this->beta, this->ngllz);

  std::cout << " Quadrature in X-dimension \n";
  std::cout << *gllx << std::endl;

  std::cout << " Quadrature in Z-dimension \n";
  std::cout << *gllz << std::endl;

  return std::make_tuple(gllx, gllz);
}

specfem::runtime_configuration::quadrature::quadrature(
    const std::string quadrature) {

  if (quadrature == "GLL4") {
    *this = specfem::runtime_configuration::quadrature(0.0, 0.0, 5, 5);
  } else if (quadrature == "GLL7") {
    *this = specfem::runtime_configuration::quadrature(0.0, 0.0, 8, 8);
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
        Node["ngllx"].as<int>(), Node["ngllz"].as<int>());
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
