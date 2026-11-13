#include "specfem/program/injection.hpp"
#include "specfem/logger.hpp"
#include "specfem/mpi.hpp"
#include <cstddef>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

specfem::injection::PluginLibrary specfem::program_impl::load_injection_plugin(
    const specfem::injection::PluginOptions &options) {
  std::optional<specfem::injection::PluginLibrary> plugin;
  std::string error;
  try {
    plugin.emplace(specfem::injection::PluginLibrary::open(options.plugin));
  } catch (const std::exception &e) {
    error = e.what();
  }

  // A plugin path can be valid on some ranks only (e.g. a node-local
  // filesystem), so agree on the outcome before any rank moves on.
  int failed = plugin ? 0 : 1;
  SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX,
                                     specfem::MPI::communicator()));
  if (failed != 0)
    throw std::runtime_error(error.empty() ? "Loading injection plugin '" +
                                                 options.plugin +
                                                 "' failed on another rank."
                                           : error);

  std::ostringstream message;
  message << "Injection plugin\n"
          << "================\n"
          << "  Name         : " << plugin->name() << "\n"
          << "  Version      : " << plugin->version() << "\n"
          << "  ABI version  : " << plugin->abi_version() << "\n"
          << "  Source       : " << plugin->source() << "\n";
  if (!plugin->sha256().empty())
    message << "  SHA-256      : " << plugin->sha256() << "\n";
  message << "  Configuration: " << options.config << "\n";
  specfem::Logger::info(message.str(), true);

  return std::move(*plugin);
}

std::shared_ptr<
    specfem::injection::InjectionMethod<specfem::element::dimension_tag::dim3>>
specfem::program_impl::make_injection_method(
    specfem::injection::PluginLibrary library,
    const specfem::injection::PluginOptions &options,
    const specfem::runtime_configuration::Injection &config,
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly,
    const int nstep, const specfem::injection::MethodRegistry &registry) {
  long long rows = assembly.injection_surface.nrows;
  SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE, &rows, 1, MPI_LONG_LONG,
                                     MPI_SUM, specfem::MPI::communicator()));
  if (rows == 0)
    throw std::runtime_error(
        "Injection plugin '" + library.name() +
        "' was given, but the mesh declares no injection faces. Mesh with "
        "STACEY_AS_INJECTION_BOUNDARY or INJECTION_XMIN/XMAX/YMIN/YMAX/"
        "BOTTOM/TOP.");

  const auto method = registry.make(config.method());

  specfem::injection::SessionOptions session_options;
  session_options.window_steps = config.window_steps();
  if (const auto budget = config.memory_budget_mib())
    session_options.memory_budget = static_cast<std::size_t>(*budget) << 20;
  if (const auto validate = config.validate())
    session_options.validate = *validate;

  specfem::injection::ProviderSession session(
      std::move(library), options.config,
      specfem::injection::make_simulation_parameters(assembly, nstep),
      session_options);
  session.setup(specfem::injection::make_injection_points(
      assembly, method->quantities(), method->name()));
  const int window_steps = session.window_steps();
  method->setup(assembly, std::move(session));

  std::ostringstream message;
  message << "Injection method\n"
          << "================\n"
          << "  Method        : " << method->name() << "\n"
          << "  Points        : " << rows << "\n"
          << "  Window (steps): " << window_steps << "\n";
  specfem::Logger::info(message.str(), true);
  return method;
}
