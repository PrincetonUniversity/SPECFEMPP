#include "parameter_parser/interface.hpp"
#include "quadrature/interface.hpp"
#include "yaml-cpp/yaml.h"
#include <tuple>

std::tuple<specfem::quadrature::quadrature *, specfem::quadrature::quadrature *>
specfem::runtime_configuration::quadrature::instantiate() {
  specfem::quadrature::quadrature *gllx =
      new specfem::quadrature::gll::gll(this->alpha, this->beta, this->ngllx);
  specfem::quadrature::quadrature *gllz =
      new specfem::quadrature::gll::gll(this->alpha, this->beta, this->ngllz);
  return std::make_tuple(gllx, gllz);
}

specfem::runtime_configuration::quadrature::quadrature(const YAML::Node &Node) {
  *this = specfem::runtime_configuration::quadrature(
      Node["alpha"].as<type_real>(), Node["beta"].as<type_real>(),
      Node["ngllx"].as<int>(), Node["ngllz"].as<int>());
}
