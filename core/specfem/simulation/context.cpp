#include "specfem/simulation/context.hpp"
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace specfem {

Context::Context(int argc, char *argv[])
    : kokkos_guard_(argc, argv),
      mpi_(std::make_unique<specfem::MPI::MPI>(&argc, &argv)) {}

Context::~Context() = default;

specfem::MPI::MPI *Context::get_mpi() const { return mpi_.get(); }

// ============================================================================
// ContextGuard Implementation
// ============================================================================

ContextGuard::ContextGuard(int argc, char *argv[])
    : context_(std::make_unique<Context>(argc, argv)) {}

ContextGuard::ContextGuard(const std::vector<std::string> &args) {
  int argc;
  char **argv;
  setup_argc_argv(args, argc, argv);

  try {
    context_ = std::make_unique<Context>(argc, argv);
  } catch (...) {
    cleanup_argc_argv(argc, argv);
    throw;
  }

  cleanup_argc_argv(argc, argv);
}

ContextGuard::~ContextGuard() = default;

Context &ContextGuard::get_context() { return *context_; }

specfem::MPI::MPI *ContextGuard::get_mpi() const { return context_->get_mpi(); }

void ContextGuard::setup_argc_argv(const std::vector<std::string> &args,
                                   int &argc, char **&argv) {
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

void ContextGuard::cleanup_argc_argv(int argc, char **argv) {
  if (argv) {
    for (int i = 0; i < argc; ++i) {
      delete[] argv[i];
    }
    delete[] argv;
  }
}

} // namespace specfem
