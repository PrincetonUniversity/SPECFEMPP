#include "specfem/program/context.hpp"
#include "specfem/logger.hpp"
#include "specfem/mpi.hpp"
#include "specfem/program/device_selection.hpp"
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace specfem::program {

Context::Context(int argc, char *argv[]) {
  // Initialize Kokkos on the intended device
  specfem::program_impl::select_default_device();
  kokkos_guard_ = std::make_unique<Kokkos::ScopeGuard>(argc, argv);
  const std::string device = specfem::program_impl::check_execution_device();
  // Initialize static MPI wrapper
  specfem::MPI::initialize(&argc, &argv);
  // Initialize Logger
  specfem::Logger::initialize(this);
  if (!device.empty()) {
    specfem::Logger::info(device);
  }
}

Context::Context(int argc, char *argv[], int nprocs) {
  // Initialize Kokkos on the intended device
  specfem::program_impl::select_default_device();
  kokkos_guard_ = std::make_unique<Kokkos::ScopeGuard>(argc, argv);
  const std::string device = specfem::program_impl::check_execution_device();
  // Initialize static MPI wrapper with user-defined processes
  specfem::MPI::initialize(&argc, &argv, nprocs);
  // Initialize Logger
  specfem::Logger::initialize(this);
  if (!device.empty()) {
    specfem::Logger::info(device);
  }
}

Context::Context(const std::vector<std::string> &args) {
  int argc;
  char **argv;
  setup_argc_argv(args, argc, argv);

  try {
    // Initialize Kokkos on the intended device
    specfem::program_impl::select_default_device();
    kokkos_guard_ = std::make_unique<Kokkos::ScopeGuard>(argc, argv);
    const std::string device = specfem::program_impl::check_execution_device();

    // Initialize static MPI wrapper
    specfem::MPI::initialize(&argc, &argv);

    // Initialize Logger
    specfem::Logger::initialize(this);
    if (!device.empty()) {
      specfem::Logger::info(device);
    }
  } catch (...) {
    cleanup_argc_argv(argc, argv);
    throw;
  }

  cleanup_argc_argv(argc, argv);
}

Context::~Context() {
  // Finalize Logger (before MPI)
  specfem::Logger::finalize();
  // Finalize static MPI wrapper
  specfem::MPI::finalize();
}

void Context::setup_argc_argv(const std::vector<std::string> &args, int &argc,
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

void Context::cleanup_argc_argv(int argc, char **argv) {
  if (argv) {
    for (int i = 0; i < argc; ++i) {
      delete[] argv[i];
    }
    delete[] argv;
  }
}

} // namespace specfem::program
