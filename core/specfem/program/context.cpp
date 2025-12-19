#include "specfem/program/context.hpp"
#include "specfem/logger.hpp"
#include "specfem/mpi.hpp"
#include "specfem_mpi/interface.hpp"
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace specfem::program {

Context::Context(int argc, char *argv[])
    : kokkos_guard_(std::make_unique<Kokkos::ScopeGuard>(argc, argv)),
      mpi_(std::make_unique<specfem::MPI::MPI>(&argc, &argv)) {
  // Initialize static MPI wrapper
  specfem::MPI_new::initialize(&argc, &argv);
  // Initialize Logger
  specfem::Logger::initialize(this);
}

Context::Context(const std::vector<std::string> &args) {
  int argc;
  char **argv;
  setup_argc_argv(args, argc, argv);

  try {
    // Initialize Kokkos
    kokkos_guard_ = std::make_unique<Kokkos::ScopeGuard>(argc, argv);

    // Initialize old MPI
    mpi_ = std::make_unique<specfem::MPI::MPI>(&argc, &argv);

    // Initialize static MPI wrapper
    specfem::MPI_new::initialize(&argc, &argv);

    // Initialize Logger
    specfem::Logger::initialize(this);
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
  specfem::MPI_new::finalize();
}

specfem::MPI::MPI *Context::get_mpi() const { return mpi_.get(); }

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
