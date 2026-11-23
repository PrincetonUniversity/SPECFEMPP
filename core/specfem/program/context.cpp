#include "specfem/program/context.hpp"
#include "specfem/logger.hpp"
#include "specfem/mpi.hpp"
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace specfem::program::context_impl {

// SYCL is configured as a GPU backend; fail early on any other SYCL device.
void check_sycl_device() {
#if defined(KOKKOS_ENABLE_SYCL)
  const auto device = Kokkos::SYCL().sycl_queue().get_device();
  if (!device.is_gpu()) {
    throw std::runtime_error(
        "SPECFEM++ SYCL builds require a GPU, but Kokkos selected '" +
        device.get_info<sycl::info::device::name>() +
        "'. Select a GPU, e.g. ONEAPI_DEVICE_SELECTOR=level_zero:gpu");
  }
#endif
}

} // namespace specfem::program::context_impl

namespace specfem::program {

Context::Context(int argc, char *argv[])
    : kokkos_guard_(std::make_unique<Kokkos::ScopeGuard>(argc, argv)) {
  specfem::program::context_impl::check_sycl_device();
  // Initialize static MPI wrapper
  specfem::MPI::initialize(&argc, &argv);
  // Initialize Logger
  specfem::Logger::initialize(this);
}

Context::Context(int argc, char *argv[], int nprocs)
    : kokkos_guard_(std::make_unique<Kokkos::ScopeGuard>(argc, argv)) {
  specfem::program::context_impl::check_sycl_device();
  // Initialize static MPI wrapper with user-defined processes
  specfem::MPI::initialize(&argc, &argv, nprocs);
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
    specfem::program::context_impl::check_sycl_device();

    // Initialize static MPI wrapper
    specfem::MPI::initialize(&argc, &argv);

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
