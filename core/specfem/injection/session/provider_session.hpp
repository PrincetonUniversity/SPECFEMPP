#pragma once

#include "specfem/injection/plugin_library.hpp"
#include "specfem/injection/request.hpp"
#include "specfem/injection/session/frame_buffer.hpp"
#include "specfem/injection/session/points.hpp"
#include "specfem/injection/session/window.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace specfem {
namespace injection {

/**
 * @brief Simulation parameters handed to the plugin when it is created.
 */
struct SimulationParameters {
  double t0 = 0.0;                   ///< Time of step 0 in s
  double dt = 0.0;                   ///< Time step in s
  int nstep = 0;                     ///< Total number of time steps
  std::array<double, 6> bounds = {}; ///< xmin, xmax, ymin, ymax, zmin, zmax
};

/**
 * @brief Options controlling how the session windows the incident field.
 */
struct SessionOptions {
  /** @brief Steps per window; chosen from memory_budget when unset. */
  std::optional<int> window_steps;
  /** @brief Bytes for the staging and device buffers together. */
  std::size_t memory_budget = std::size_t{ 256 } << 20;
#ifdef NDEBUG
  bool validate = false; ///< Reject non-finite values after each fill
#else
  bool validate = true; ///< Reject non-finite values after each fill
#endif
};

/**
 * @brief Drives an injection plugin and serves its incident field on the
 * device, one window of time steps at a time.
 *
 * Every call is collective: all ranks construct the session, call setup, and
 * request the same windows, including ranks without injection points. If the
 * plugin fails on any rank, every rank throws.
 *
 * Windows are aligned to multiples of window_steps(), so the incident field
 * served for a step does not depend on the order in which steps are
 * requested.
 *
 * @code
 * specfem::injection::ProviderSession session(std::move(plugin), config,
 *                                             parameters);
 * session.setup(points, request);
 * for (int istep = 0; istep < nstep; ++istep) {
 *   session.ensure_window(istep);
 *   const auto frame = session.frame(); // read with frame.load(istep, ...)
 * }
 * @endcode
 */
class ProviderSession {
public:
  /**
   * @brief Create the plugin instance.
   *
   * @param library Loaded plugin
   * @param config_path Plugin configuration file, passed to the plugin as is
   * @param parameters Simulation parameters
   * @param options Windowing and validation options
   * @throws std::runtime_error on every rank if creation fails on any rank
   */
  ProviderSession(specfem::injection::PluginLibrary library,
                  const std::string &config_path,
                  const SimulationParameters &parameters,
                  const SessionOptions &options = {});

  /**
   * @brief Hand the injection points and the request to the plugin, and
   * allocate the buffers.
   *
   * Checks that the plugin supports every requested quantity first, and lists
   * all unsupported ones if it does not.
   *
   * @param points Injection points on this rank (may be empty)
   * @param request Quantities requested per medium
   * @throws std::runtime_error on every rank if the check or the plugin's
   * setup fails on any rank
   */
  void setup(const InjectionPoints &points, const Request &request);

  /**
   * @brief Make sure the device window contains a step, filling it if needed.
   *
   * @param istep Simulation step in [0, nstep)
   * @throws std::runtime_error on every rank if the plugin's fill fails, or
   * returns non-finite values while validation is on, on any rank
   */
  void ensure_window(const int istep);

  /**
   * @brief The current device window.
   * @return Frame buffer, valid until the next ensure_window
   */
  [[nodiscard]] const FrameBuffer &frame() const { return frame_; }

  /**
   * @brief Number of steps per window, agreed across ranks.
   * @return Window size; 0 before setup
   */
  [[nodiscard]] int window_steps() const { return window_steps_; }

  /**
   * @brief The plugin driven by this session.
   * @return Loaded plugin
   */
  [[nodiscard]] const specfem::injection::PluginLibrary &library() const {
    return library_;
  }

private:
  /** @brief Host staging buffer the plugin writes into. */
#ifdef KOKKOS_HAS_SHARED_HOST_PINNED_SPACE
  using staging_space = Kokkos::SharedHostPinnedSpace;
#else
  using staging_space = Kokkos::HostSpace;
#endif
  using staging_view = Kokkos::View<type_real ***, Kokkos::LayoutRight,
                                    staging_space>; ///< [row][slot][step]

  void choose_window_steps();
  std::optional<std::string> fill_window(const int start, const int nsteps);
  std::optional<std::string> validate_window(const int start,
                                             const int nsteps) const;

  specfem::injection::PluginLibrary library_; ///< Loaded plugin
  std::optional<specfem::injection::PluginInstance> instance_; ///< Plugin
                                                               ///< instance
  SimulationParameters parameters_;  ///< Simulation parameters
  SessionOptions options_;           ///< Windowing and validation options
  Request request_;                  ///< Requested quantities
  std::vector<std::int32_t> medium_; ///< Medium of each row
  std::int64_t nrows_ = 0;           ///< Injection points on this rank
  int nslots_ = 0;                   ///< Slots per row
  int window_steps_ = 0;             ///< Steps per window
  staging_view staging_;             ///< Host buffer filled by the plugin
  FrameBuffer frame_;                ///< Device copy of the current window
};

} // namespace injection
} // namespace specfem
