#include "execute.hpp"

int main(int argc, char **argv) {
  // Initialize MPI
  specfem::MPI::MPI *mpi = new specfem::MPI::MPI(&argc, &argv);
  // Initialize Kokkos
  Kokkos::initialize(argc, argv);
  {
    const std::string default_file = __default_file__;
    const std::string parameters_file = __parameters_file__;
    const YAML::Node parameter_dict = YAML::LoadFile(parameters_file);
    const YAML::Node default_dict = YAML::LoadFile(default_file);
    std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task> > tasks;
    execute(parameter_dict, default_dict, tasks, mpi);
  }
  // Finalize Kokkos
  Kokkos::finalize();
  // Finalize MPI
  delete mpi;
  return 0;
}
