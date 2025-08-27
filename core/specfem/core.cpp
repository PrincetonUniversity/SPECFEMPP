#include "specfem/core.hpp"
#include "specfem/execute.hpp"
#include <cstring>
#include <iostream>

namespace specfem {

// Static member initialization
Core *Core::instance_ = nullptr;

Core &Core::instance() {
  if (instance_ == nullptr) {
    instance_ = new Core();
  }
  return *instance_;
}

Core::Core()
    : mpi_(nullptr), kokkos_initialized_(false), mpi_initialized_(false),
      core_initialized_(false) {}

Core::~Core() {
  if (core_initialized_) {
    finalize();
  }
}

bool Core::initialize(int argc, char *argv[]) {
  if (core_initialized_) {
    return false; // Already initialized
  }

  try {
    // Initialize MPI first
    mpi_ = new specfem::MPI::MPI(&argc, &argv);
    mpi_initialized_ = true;

    // Initialize Kokkos
    Kokkos::initialize(argc, argv);
    kokkos_initialized_ = true;

    core_initialized_ = true;
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Error during initialization: " << e.what() << std::endl;
    // Cleanup on failure
    if (kokkos_initialized_) {
      Kokkos::finalize();
      kokkos_initialized_ = false;
    }
    if (mpi_initialized_ && mpi_) {
      delete mpi_;
      mpi_ = nullptr;
      mpi_initialized_ = false;
    }
    return false;
  }
}

bool Core::initialize_from_python(const std::vector<std::string> &py_argv) {
  if (core_initialized_) {
    return false; // Already initialized
  }

  int argc;
  char **argv;
  setup_argc_argv(py_argv, argc, argv);

  bool result = initialize(argc, argv);

  cleanup_argc_argv(argc, argv);
  return result;
}

bool Core::finalize() {
  if (!core_initialized_) {
    return false;
  }

  try {
    // Finalize Kokkos
    if (kokkos_initialized_) {
      Kokkos::finalize();
      kokkos_initialized_ = false;
    }

    // Finalize MPI
    if (mpi_initialized_ && mpi_) {
      delete mpi_;
      mpi_ = nullptr;
      mpi_initialized_ = false;
    }

    core_initialized_ = false;
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Error during finalization: " << e.what() << std::endl;
    return false;
  }
}

template <specfem::dimension::type DIM>
bool Core::execute(
    const YAML::Node &parameter_dict, const YAML::Node &default_dict,
    std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task> >
        &tasks) {
  if (!core_initialized_) {
    std::cerr << "Core not initialized. Call initialize() first." << std::endl;
    return false;
  }

  try {
    // Call the appropriate templated execute function
    if constexpr (DIM == specfem::dimension::type::dim2) {
      ::execute(parameter_dict, default_dict, tasks, mpi_);
    } else if constexpr (DIM == specfem::dimension::type::dim3) {
      // For now, call the same execute function - this will need to be
      // specialized once we have proper dimension-templated execute functions
      ::execute(parameter_dict, default_dict, tasks, mpi_);
    } else {
      std::cerr << "Unsupported dimension type" << std::endl;
      return false;
    }
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Error during execution: " << e.what() << std::endl;
    return false;
  }
}

bool Core::execute_with_dimension(
    const std::string &dimension, const YAML::Node &parameter_dict,
    const YAML::Node &default_dict,
    std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task> >
        &tasks) {
  if (dimension == "2d" || dimension == "dim2") {
    return execute<specfem::dimension::type::dim2>(parameter_dict, default_dict,
                                                   tasks);
  } else if (dimension == "3d" || dimension == "dim3") {
    return execute<specfem::dimension::type::dim3>(parameter_dict, default_dict,
                                                   tasks);
  } else {
    std::cerr << "Invalid dimension: " << dimension << ". Use '2d' or '3d'."
              << std::endl;
    return false;
  }
}

specfem::MPI::MPI *Core::get_mpi() const { return mpi_; }

bool Core::is_initialized() const { return core_initialized_; }

bool Core::is_kokkos_initialized() const { return kokkos_initialized_; }

void Core::setup_argc_argv(const std::vector<std::string> &args, int &argc,
                           char **&argv) {
  argc = args.size();
  argv = new char *[argc + 1];

  for (size_t i = 0; i < args.size(); ++i) {
    const std::string &str = args[i];
    argv[i] = new char[str.length() + 1];
    std::strcpy(argv[i], str.c_str());
  }

  // Null-terminate argv following the specification
  argv[argc] = nullptr;
}

void Core::cleanup_argc_argv(int argc, char **argv) {
  if (argv) {
    for (int i = 0; i < argc; ++i) {
      delete[] argv[i];
    }
    delete[] argv;
  }
}

// Explicit template instantiations
template bool Core::execute<specfem::dimension::type::dim2>(
    const YAML::Node &, const YAML::Node &,
    std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task> > &);

template bool Core::execute<specfem::dimension::type::dim3>(
    const YAML::Node &, const YAML::Node &,
    std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task> > &);

} // namespace specfem
