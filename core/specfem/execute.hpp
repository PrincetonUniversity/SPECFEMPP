#pragma once

#include "specfem/assembly.hpp"
// #include "coupled_interface/interface.hpp"
// #include "domain/interface.hpp"
#include "enumerations/dimension.hpp"
#include "io/interface.hpp"
#include "kokkos_abstractions.h"
#include "mesh/mesh.hpp"
#include "parameter_parser/interface.hpp"
#include "solver/solver.hpp"
#include "specfem/receivers.hpp"
#include "specfem/source.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include "timescheme/timescheme.hpp"
#include "yaml-cpp/yaml.h"
#include <Kokkos_Core.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <ctime>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Legacy execute function (currently used by existing code)
void execute(
    const YAML::Node &parameter_dict, const YAML::Node &default_dict,
    std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task> > tasks,
    specfem::MPI::MPI *mpi);

// Future templated execute functions (to be implemented)
// These will replace the legacy function once dimension-specific logic is
// refactored

template <specfem::dimension::type DIM>
void execute_templated(
    const YAML::Node &parameter_dict, const YAML::Node &default_dict,
    std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task> > tasks,
    specfem::MPI::MPI *mpi);

// Explicit declarations for template specializations
extern template void execute_templated<specfem::dimension::type::dim2>(
    const YAML::Node &, const YAML::Node &,
    std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task> >,
    specfem::MPI::MPI *);

extern template void execute_templated<specfem::dimension::type::dim3>(
    const YAML::Node &, const YAML::Node &,
    std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task> >,
    specfem::MPI::MPI *);
