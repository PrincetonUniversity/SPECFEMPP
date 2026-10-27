#include "specfem/fk_plugin/impl/kokkos_lifetime.hpp"

#include <Kokkos_Core.hpp>

#include <mutex>
#include <stdexcept>

struct specfem::fk_plugin_impl::KokkosLifetime::State {
  std::mutex mutex;
  int instances = 0;
  bool owns_kokkos = false; ///< Whether a guard called Kokkos::initialize
};

specfem::fk_plugin_impl::KokkosLifetime::State &
specfem::fk_plugin_impl::KokkosLifetime::state() {
  static State instance;
  return instance;
}

specfem::fk_plugin_impl::KokkosLifetime::KokkosLifetime() {
  auto &shared = state();
  const std::lock_guard<std::mutex> lock(shared.mutex);
  if (shared.instances == 0) {
    if (Kokkos::is_finalized())
      throw std::runtime_error(
          "FK plugin: its Kokkos was already finalized; the plugin cannot be "
          "created again in this process");
    if (!Kokkos::is_initialized()) {
      Kokkos::initialize();
      shared.owns_kokkos = true;
    }
  }
  ++shared.instances;
}

specfem::fk_plugin_impl::KokkosLifetime::~KokkosLifetime() {
  auto &shared = state();
  const std::lock_guard<std::mutex> lock(shared.mutex);
  if (--shared.instances == 0 && shared.owns_kokkos) {
    Kokkos::finalize();
    shared.owns_kokkos = false;
  }
}
