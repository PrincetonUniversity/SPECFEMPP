#include "specfem/program/injection.hpp"
#include "specfem/logger.hpp"
#include "specfem/mpi.hpp"
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

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
          << "  Source       : " << plugin->source() << "\n"
          << "  Configuration: " << options.config << "\n";
  specfem::Logger::info(message.str(), true);
  specfem::Logger::warning(
      "Injection plugin loaded, but injection methods are not implemented "
      "yet; the incident field is not applied in this run.",
      true);

  return std::move(*plugin);
}
