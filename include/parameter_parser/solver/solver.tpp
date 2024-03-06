#ifndef _SPECFEM_RUNTIME_CONFIGURATION_SOLVER_SOLVER_TPP_
#define _SPECFEM_RUNTIME_CONFIGURATION_SOLVER_SOLVER_TPP_

#include "kernels/kernels.hpp"
#include "solver.hpp"
#include "solver/time_marching.hpp"
#include "timescheme/interface.hpp"
#include <iostream>
#include <memory>

template <typename qp_type>
std::shared_ptr<specfem::solver::solver>
specfem::runtime_configuration::solver::solver::instantiate(
    const specfem::compute::assembly &assembly,
    std::shared_ptr<specfem::time_scheme::time_scheme> time_scheme,
    const qp_type &quadrature) const {

  constexpr auto forward = specfem::simulation::type::forward;
  constexpr auto dimension = specfem::dimension::type::dim2;

  if (this->simulation_type == "forward") {
    std::cout << "Instantiating Kernels \n";
    std::cout << "-------------------------------\n";
    const auto kernels = specfem::kernels::kernels<forward, dimension, qp_type>(
        assembly, quadrature);
    return std::make_shared<
        specfem::solver::time_marching<forward, dimension, qp_type> >(
        kernels, time_scheme);
  } else {
    throw std::runtime_error("Simulation type not implemented");
  }
}

#endif
