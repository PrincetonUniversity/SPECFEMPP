
#include "specfem/source.hpp"

#include "specfem_setup.hpp"
#include <cmath>
#include <yaml-cpp/yaml.h>

template <specfem::element::dimension_tag DimensionTag>
template <specfem::element::dimension_tag U, typename std::enable_if<U == specfem::element::dimension_tag::dim2>::type*>
specfem::sources::source<DimensionTag>::source(
    YAML::Node &Node, const int nsteps, const type_real dt)
    : global_coordinates(Node["x"].as<type_real>(), Node["z"].as<type_real>()) {

  // Read source time function
  this->set_source_time_function(Node, nsteps, dt);

  return;
}
