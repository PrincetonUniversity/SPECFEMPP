#pragma once

#include "specfem/compute.hpp"
#include "specfem/solver.hpp"
#include "specfem/timescheme/newmark.hpp"
#include <iostream>
#include <memory>

template <int NGLL, specfem::element::dimension_tag DimensionTag>
std::shared_ptr<specfem::solver::solver>
specfem::runtime_configuration::solver::solver::instantiate(
    const type_real dt, const specfem::assembly::assembly<DimensionTag> &assembly,
    std::shared_ptr<specfem::time_scheme::time_scheme> time_scheme,
    const std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task<DimensionTag>>> &tasks)
    const {

  if (specfem::utilities::is_forward_string(this->simulation_type)) {
    return std::make_shared<
        specfem::solver::time_marching<specfem::simulation::type::forward,
                                       DimensionTag, NGLL> >(
        time_scheme, tasks, assembly);
  } else if (specfem::utilities::is_combined_string(this->simulation_type)) {

    if (DimensionTag == specfem::element::dimension_tag::dim3) {
      throw std::runtime_error(
          "Combined simulation not implemented for 3D problems");
    }

    return std::make_shared<
        specfem::solver::time_marching<specfem::simulation::type::combined,
                                       DimensionTag, NGLL> >(
        assembly, time_scheme, tasks);
  } else {
    throw std::runtime_error("Simulation type not recognized");
  }
}
