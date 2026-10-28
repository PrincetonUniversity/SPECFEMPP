#pragma once

namespace specfem {
namespace fk_plugin_impl {

/**
 * @brief Keeps the plugin's private Kokkos initialized while any plugin
 * instance exists.
 *
 * The first instance initializes Kokkos unless it is already running; the
 * last one finalizes it, but only if this guard initialized it. Kokkos
 * cannot be initialized again after finalize, so constructing a guard after
 * that throws.
 */
class KokkosLifetime {
public:
  /**
   * @brief Register an instance, initializing Kokkos if needed.
   * @throws std::runtime_error if Kokkos was already finalized
   */
  KokkosLifetime();

  /** @brief Unregister; the last instance finalizes Kokkos if it owns it. */
  ~KokkosLifetime();

  KokkosLifetime(const KokkosLifetime &) = delete;
  KokkosLifetime &operator=(const KokkosLifetime &) = delete;
  KokkosLifetime(KokkosLifetime &&) = delete;
  KokkosLifetime &operator=(KokkosLifetime &&) = delete;

private:
  struct State;
  /** @brief The module-wide instance count, in a function-local static. */
  static State &state();
};

} // namespace fk_plugin_impl
} // namespace specfem
