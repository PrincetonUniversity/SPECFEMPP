#include "solver.hpp"
#include "solver.tpp"

template std::shared_ptr<specfem::solver::solver>
specfem::runtime_configuration::solver::instantiate<
    5, specfem::element::dimension_tag::dim2>(
    const type_real,
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim2> &,
    std::shared_ptr<specfem::time_scheme::time_scheme>,
    const specfem::simulation::type,
    const std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task<
        specfem::element::dimension_tag::dim2>>> &) const;

template std::shared_ptr<specfem::solver::solver>
specfem::runtime_configuration::solver::instantiate<
    5, specfem::element::dimension_tag::dim3>(
    const type_real,
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3> &,
    std::shared_ptr<specfem::time_scheme::time_scheme>,
    const specfem::simulation::type,
    const std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task<
        specfem::element::dimension_tag::dim3>>> &) const;
