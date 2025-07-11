#include "specfem/assembly.hpp"
// #include "coupled_interface/interface.hpp"
// #include "domain/interface.hpp"
#include "io/interface.hpp"
#include "kokkos_abstractions.h"
#include "mesh/mesh.hpp"
#include "parameter_parser/interface.hpp"
#include "solver/solver.hpp"
#include "source/interface.hpp"
#include "specfem/receivers.hpp"
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
// Specfem2d driver

void execute(
    const YAML::Node &parameter_dict, const YAML::Node &default_dict,
    std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task> > tasks,
    specfem::MPI::MPI *mpi);
