
#include "specfem/source.hpp"
#include "kokkos_abstractions.h"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include <yaml-cpp/yaml.h>
#include <cmath>


template <specfem::dimension::type DimensionTag>
template <specfem::dimension::type U, typename std::enable_if<U == specfem::dimension::type::dim2>::type*>
specfem::sources::source<DimensionTag>::source(
    YAML::Node &Node, const int nsteps, const type_real dt)
    : global_coordinates(Node["x"].as<type_real>(), Node["z"].as<type_real>()) {

  // Read source time function
  this->set_forcing_function(Node, nsteps, dt);

  return;
}
