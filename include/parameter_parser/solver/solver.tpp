#ifndef _SPECFEM_RUNTIME_CONFIGURATION_SOLVER_SOLVER_TPP_
#define _SPECFEM_RUNTIME_CONFIGURATION_SOLVER_SOLVER_TPP_

#include "kokkos_kernels/domain_kernels.hpp"
#include "solver.hpp"
#include "solver/time_marching.hpp"
#include "timescheme/newmark.hpp"
#include <iostream>
#include <memory>

template <int NGLL>
std::shared_ptr<specfem::solver::solver>
specfem::runtime_configuration::solver::solver::instantiate(
    const type_real dt, const specfem::compute::assembly &assembly,
    std::shared_ptr<specfem::time_scheme::time_scheme> time_scheme,
    const std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task> > &tasks)
    const {

  if (this->simulation_type == "forward") {
    std::cout << "Instantiating Kernels \n";
    std::cout << "-------------------------------\n";
    const auto kernels =
        specfem::kokkos_kernels::domain_kernels<specfem::wavefield::simulation_field::forward,
                                  specfem::dimension::type::dim2, NGLL>(
            assembly);
    return std::make_shared<
        specfem::solver::time_marching<specfem::simulation::type::forward,
                                       specfem::dimension::type::dim2, NGLL> >(
        kernels, time_scheme, tasks);
  } else if (this->simulation_type == "combined") {
    std::cout << "Instantiating Kernels \n";
    std::cout << "-------------------------------\n";
    const auto adjoint_kernels =
        specfem::kokkos_kernels::domain_kernels<specfem::wavefield::simulation_field::adjoint,
                                  specfem::dimension::type::dim2, NGLL>(
            assembly);
    const auto backward_kernels = specfem::kokkos_kernels::domain_kernels<
        specfem::wavefield::simulation_field::backward,
        specfem::dimension::type::dim2, NGLL>(assembly);
    return std::make_shared<
        specfem::solver::time_marching<specfem::simulation::type::combined,
                                       specfem::dimension::type::dim2, NGLL> >(
        assembly, adjoint_kernels, backward_kernels, time_scheme, tasks);
  } else {
    throw std::runtime_error("Simulation type not recognized");
  }
}

#endif
