#pragma once

#include "specfem/plugin/injection_abi.h"
#include <array>
#include <cassert>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace specfem {
/**
 * @brief C++ layer of the injection plugin SDK.
 *
 * Plugin authors derive from @c InjectionPlugin and register the class with
 * @c SPECFEMPP_DEFINE_INJECTION_PLUGIN; the macro implements the C ABI in
 * @c specfem/plugin/injection_abi.h on top of it.
 */
namespace plugin {

/** @brief Medium of an injection point. */
enum class Medium : std::int32_t {
  elastic = spfp_medium_elastic,  ///< Elastic solid
  acoustic = spfp_medium_acoustic ///< Acoustic fluid
};

/** @brief Physical quantity SPECFEM++ can request from a plugin. */
enum class Quantity : std::int32_t {
  displacement = spfp_quantity_displacement,    ///< Displacement (3)
  velocity = spfp_quantity_velocity,            ///< Velocity (3)
  acceleration = spfp_quantity_acceleration,    ///< Acceleration (3)
  traction = spfp_quantity_traction,            ///< Traction on the normal (3)
  potential_rate = spfp_quantity_potential_rate ///< Fluid potential rate (1)
};

/**
 * @brief Number of buffer components a quantity occupies.
 *
 * @param quantity Requested quantity
 * @return 1 for @c potential_rate, 3 for vector quantities
 */
constexpr int component_count(const Quantity quantity) {
  return quantity == Quantity::potential_rate ? 1 : 3;
}

/**
 * @brief Whether a raw ABI value names a known medium.
 *
 * @param medium Raw medium value
 * @return true for elastic or acoustic
 */
constexpr bool is_valid_medium(const spfp_medium medium) {
  return medium == spfp_medium_elastic || medium == spfp_medium_acoustic;
}

/**
 * @brief Whether a raw ABI value names a known quantity.
 *
 * @param quantity Raw quantity value
 * @return true for every quantity defined in ABI v1
 */
constexpr bool is_valid_quantity(const spfp_quantity quantity) {
  return quantity >= spfp_quantity_displacement &&
         quantity <= spfp_quantity_potential_rate;
}

/** @brief Simulation parameters passed to the plugin at creation. */
class SimulationInfo {
public:
  /**
   * @brief Wrap the ABI struct.
   * @param info Simulation parameters from SPECFEM++
   */
  explicit SimulationInfo(const spfp_simulation_info &info) : info_(info) {}

  /** @brief Spatial dimension (2 or 3). @return dimension */
  int dimension() const { return info_.dimension; }

  /** @brief Total number of time steps. @return step count */
  int number_of_steps() const { return info_.nstep; }

  /** @brief Time of step 0 in s. @return t0 */
  double t0() const { return info_.t0; }

  /** @brief Time step in s. @return dt */
  double dt() const { return info_.dt; }

  /**
   * @brief Time of a simulation step.
   * @param step Simulation step index
   * @return @f$ t_0 + \mathrm{step}\,\Delta t @f$ in s
   */
  double time(const int step) const { return info_.t0 + step * info_.dt; }

  /**
   * @brief Domain bounding box.
   * @return xmin, xmax, ymin, ymax, zmin, zmax in m
   */
  std::array<double, 6> bounds() const {
    return { info_.bounds[0], info_.bounds[1], info_.bounds[2],
             info_.bounds[3], info_.bounds[4], info_.bounds[5] };
  }

private:
  spfp_simulation_info info_; ///< Copy of the ABI struct
};

/** @brief MPI communicator passed to the plugin at creation. */
class Communicator {
public:
  /**
   * @brief Wrap the ABI struct.
   * @param communicator Communicator description from SPECFEM++
   */
  explicit Communicator(const spfp_communicator &communicator)
      : communicator_(communicator) {}

  /** @brief Whether SPECFEM++ runs with MPI. @return true with MPI */
  bool has_mpi() const { return communicator_.has_mpi != 0; }

  /** @brief Rank of this process. @return rank */
  int rank() const { return communicator_.rank; }

  /** @brief Number of ranks. @return communicator size */
  int size() const { return communicator_.size; }

  /**
   * @brief Fortran handle of the communicator; convert with MPI_Comm_f2c.
   * @return MPI_Comm_c2f(comm)
   */
  std::int32_t fortran_handle() const { return communicator_.fortran_handle; }

private:
  spfp_communicator communicator_; ///< Copy of the ABI struct
};

/**
 * @brief Read-only view of the injection points, valid only during setup.
 *
 * Copy whatever the plugin needs; the arrays are released after @c setup
 * returns.
 */
class Points {
public:
  /**
   * @brief Wrap the ABI struct.
   * @param points Points from SPECFEM++
   */
  explicit Points(const spfp_points &points) : points_(points) {}

  /** @brief Number of points. @return point count */
  std::int64_t size() const { return points_.n; }

  /** @brief x-coordinate in m. @param i point @return x */
  double x(const std::int64_t i) const { return points_.x[i]; }
  /** @brief y-coordinate in m. @param i point @return y */
  double y(const std::int64_t i) const { return points_.y[i]; }
  /** @brief z-coordinate in m (up). @param i point @return z */
  double z(const std::int64_t i) const { return points_.z[i]; }
  /** @brief Unit normal, x. @param i point @return nx */
  double nx(const std::int64_t i) const { return points_.nx[i]; }
  /** @brief Unit normal, y. @param i point @return ny */
  double ny(const std::int64_t i) const { return points_.ny[i]; }
  /** @brief Unit normal, z. @param i point @return nz */
  double nz(const std::int64_t i) const { return points_.nz[i]; }
  /** @brief Medium of the point. @param i point @return medium */
  Medium medium(const std::int64_t i) const {
    return static_cast<Medium>(points_.medium[i]);
  }
  /** @brief Local density. @param i point @return rho in kg/m^3 */
  double rho(const std::int64_t i) const { return points_.rho[i]; }
  /** @brief Local P speed. @param i point @return vp in m/s */
  double vp(const std::int64_t i) const { return points_.vp[i]; }
  /** @brief Local S speed (0 in fluids). @param i point @return vs in m/s */
  double vs(const std::int64_t i) const { return points_.vs[i]; }

private:
  spfp_points points_; ///< Copy of the ABI struct (arrays not owned)
};

/**
 * @brief Quantities requested per medium and their component layout.
 *
 * For a point of medium @c m, buffer components are the quantities of
 * @c quantities(m) in order, each taking @c component_count(q) components.
 */
class Request {
public:
  Request() = default;

  /**
   * @brief Build a request from per-medium quantity lists.
   * @param elastic Quantities for elastic points
   * @param acoustic Quantities for acoustic points
   */
  Request(std::vector<Quantity> elastic, std::vector<Quantity> acoustic)
      : elastic_(std::move(elastic)), acoustic_(std::move(acoustic)) {}

  /**
   * @brief Copy and validate the ABI request.
   * @param request Request from SPECFEM++
   * @throws std::invalid_argument on negative counts, null lists, or unknown
   * quantities
   */
  explicit Request(const spfp_request &request)
      : elastic_(copy(request.n_elastic, request.elastic)),
        acoustic_(copy(request.n_acoustic, request.acoustic)) {}

  /**
   * @brief Quantities requested for a medium.
   * @param medium Medium
   * @return ordered quantity list
   */
  const std::vector<Quantity> &quantities(const Medium medium) const {
    return medium == Medium::elastic ? elastic_ : acoustic_;
  }

  /**
   * @brief Total components per point of a medium.
   * @param medium Medium
   * @return sum of the component counts of its quantities
   */
  int component_count(const Medium medium) const {
    int count = 0;
    for (const auto quantity : quantities(medium))
      count += specfem::plugin::component_count(quantity);
    return count;
  }

  /**
   * @brief First buffer component of a quantity for a medium.
   * @param medium Medium
   * @param quantity Quantity
   * @return component offset, or -1 if @p quantity is not requested
   */
  int slot_offset(const Medium medium, const Quantity quantity) const {
    int offset = 0;
    for (const auto requested : quantities(medium)) {
      if (requested == quantity)
        return offset;
      offset += specfem::plugin::component_count(requested);
    }
    return -1;
  }

private:
  static std::vector<Quantity> copy(const std::int32_t count,
                                    const spfp_quantity *list) {
    if (count < 0 || (count > 0 && list == nullptr))
      throw std::invalid_argument("request: invalid quantity list");
    std::vector<Quantity> result;
    result.reserve(count);
    for (std::int32_t i = 0; i < count; ++i) {
      if (!is_valid_quantity(list[i]))
        throw std::invalid_argument("request: unknown quantity " +
                                    std::to_string(list[i]));
      result.push_back(static_cast<Quantity>(list[i]));
    }
    return result;
  }

  std::vector<Quantity> elastic_;  ///< Quantities for elastic points
  std::vector<Quantity> acoustic_; ///< Quantities for acoustic points
};

/**
 * @brief Window of the host buffer the plugin fills, valid only during fill.
 *
 * Step @c s of the window is simulation step @c first_step + s of the
 * @c fill call.
 */
class Buffer {
public:
  /**
   * @brief Wrap and validate the ABI buffer.
   * @param buffer Buffer from SPECFEM++
   * @throws std::invalid_argument on null data or an unknown dtype
   */
  explicit Buffer(const spfp_buffer &buffer) : buffer_(buffer) {
    if (buffer_.data == nullptr && buffer_.npoints > 0)
      throw std::invalid_argument("buffer: null data");
    if (buffer_.dtype != spfp_dtype_float32 &&
        buffer_.dtype != spfp_dtype_float64)
      throw std::invalid_argument("buffer: unknown dtype " +
                                  std::to_string(buffer_.dtype));
  }

  /** @brief Number of points. @return point count */
  std::int64_t number_of_points() const { return buffer_.npoints; }

  /** @brief Components per point. @return component count */
  int number_of_components() const { return buffer_.ncomponents; }

  /** @brief Steps in this window. @return step count */
  int number_of_steps() const { return buffer_.nsteps; }

  /**
   * @brief Write one value, converting to the buffer's dtype.
   * @param point Point index
   * @param component Component index
   * @param step Step index within the window
   * @param value Value to store
   */
  void set(const std::int64_t point, const int component, const int step,
           const double value) const {
    assert(point >= 0 && point < buffer_.npoints);
    assert(component >= 0 && component < buffer_.ncomponents);
    assert(step >= 0 && step < buffer_.nsteps);
    const std::int64_t offset = point * buffer_.stride_point +
                                component * buffer_.stride_component +
                                step * buffer_.stride_step;
    if (buffer_.dtype == spfp_dtype_float32)
      static_cast<float *>(buffer_.data)[offset] = static_cast<float>(value);
    else
      static_cast<double *>(buffer_.data)[offset] = value;
  }

private:
  spfp_buffer buffer_; ///< Copy of the ABI struct (data not owned)
};

/**
 * @brief Abstract base class for C++ injection plugins.
 *
 * Derive from this class, add a constructor taking
 * @c (const std::string &config_path, const SimulationInfo &, const
 * Communicator &), and register it with
 * @c SPECFEMPP_DEFINE_INJECTION_PLUGIN. Exceptions thrown from the constructor
 * or any member are converted to ABI status codes and error messages.
 *
 * @code
 * class MyPlugin : public specfem::plugin::InjectionPlugin {
 * public:
 *   MyPlugin(const std::string &config_path,
 *            const specfem::plugin::SimulationInfo &info,
 *            const specfem::plugin::Communicator &communicator);
 *   bool supports(specfem::plugin::Medium medium,
 *                 specfem::plugin::Quantity quantity) const override;
 *   void setup(const specfem::plugin::Points &points,
 *              const specfem::plugin::Request &request) override;
 *   void fill(int first_step, specfem::plugin::Buffer &buffer) override;
 * };
 * SPECFEMPP_DEFINE_INJECTION_PLUGIN("my-plugin", "1.0.0", MyPlugin)
 * @endcode
 */
class InjectionPlugin {
public:
  virtual ~InjectionPlugin() = default;

  /**
   * @brief Whether the plugin can provide a quantity for a medium.
   * @param medium Medium of the points
   * @param quantity Requested quantity
   * @return true if supported
   */
  virtual bool supports(Medium medium, Quantity quantity) const = 0;

  /**
   * @brief Receive the injection points and the request (called once).
   *
   * Copy what is needed: @p points is only valid during this call.
   *
   * @param points Injection points
   * @param request Quantities requested per medium
   */
  virtual void setup(const Points &points, const Request &request) = 0;

  /**
   * @brief Fill one window of time steps.
   *
   * Must give the same values regardless of window order or size.
   *
   * @param first_step Simulation step of window step 0
   * @param buffer Window to fill
   */
  virtual void fill(int first_step, Buffer &buffer) = 0;
};

} // namespace plugin

namespace plugin_impl {

/** @brief Instance handle passed through the C ABI as @c void*. */
struct Handle {
  std::unique_ptr<specfem::plugin::InjectionPlugin> plugin; ///< Instance
  std::string last_error;                                   ///< Last error
};

/**
 * @brief Message of the last failed create on this thread.
 * @return reference to the thread-local message
 */
inline std::string &create_error() {
  static thread_local std::string message;
  return message;
}

/**
 * @brief Run @p body, converting exceptions into a status and message.
 * @tparam Body Callable returning nothing
 * @param handle Instance whose error message is updated
 * @param body Work to run
 * @return spfp_status_ok, or spfp_status_error with handle->last_error set
 */
template <typename Body>
spfp_status guarded(Handle &handle, const Body &body) noexcept {
  try {
    body();
    handle.last_error.clear();
    return spfp_status_ok;
  } catch (const std::exception &error) {
    handle.last_error = error.what();
  } catch (...) {
    handle.last_error = "unknown exception";
  }
  return spfp_status_error;
}

/**
 * @brief ABI create: construct @p PluginType from the creation arguments.
 * @tparam PluginType Concrete plugin class
 * @param config_path Path of the plugin configuration file
 * @param info Simulation parameters
 * @param communicator MPI communicator description
 * @return new handle, or nullptr on failure (see create_error)
 */
template <typename PluginType>
void *create(const char *config_path, const spfp_simulation_info *info,
             const spfp_communicator *communicator) noexcept {
  create_error().clear();
  try {
    if (config_path == nullptr || info == nullptr || communicator == nullptr)
      throw std::invalid_argument("create: null argument");
    auto handle = std::make_unique<Handle>();
    handle->plugin = std::make_unique<PluginType>(
        std::string(config_path), specfem::plugin::SimulationInfo(*info),
        specfem::plugin::Communicator(*communicator));
    return handle.release();
  } catch (const std::exception &error) {
    create_error() = error.what();
  } catch (...) {
    create_error() = "unknown exception";
  }
  return nullptr;
}

/**
 * @brief ABI supports.
 * @param self Handle from create
 * @param medium Raw medium
 * @param quantity Raw quantity
 * @return 1 if supported, else 0 (also for invalid arguments or exceptions)
 */
inline std::int32_t supports(void *self, const spfp_medium medium,
                             const spfp_quantity quantity) noexcept {
  if (self == nullptr || !specfem::plugin::is_valid_medium(medium) ||
      !specfem::plugin::is_valid_quantity(quantity))
    return 0;
  auto &handle = *static_cast<Handle *>(self);
  bool result = false;
  const auto status = guarded(handle, [&] {
    result = handle.plugin->supports(
        static_cast<specfem::plugin::Medium>(medium),
        static_cast<specfem::plugin::Quantity>(quantity));
  });
  return (status == spfp_status_ok && result) ? 1 : 0;
}

/**
 * @brief ABI setup.
 * @param self Handle from create
 * @param points Injection points
 * @param request Per-medium request
 * @return status
 */
inline spfp_status setup(void *self, const spfp_points *points,
                         const spfp_request *request) noexcept {
  if (self == nullptr)
    return spfp_status_error;
  auto &handle = *static_cast<Handle *>(self);
  return guarded(handle, [&] {
    if (points == nullptr || request == nullptr)
      throw std::invalid_argument("setup: null argument");
    handle.plugin->setup(specfem::plugin::Points(*points),
                         specfem::plugin::Request(*request));
  });
}

/**
 * @brief ABI fill.
 * @param self Handle from create
 * @param first_step Simulation step of window step 0
 * @param buffer Window to fill
 * @return status
 */
inline spfp_status fill(void *self, const std::int32_t first_step,
                        const spfp_buffer *buffer) noexcept {
  if (self == nullptr)
    return spfp_status_error;
  auto &handle = *static_cast<Handle *>(self);
  return guarded(handle, [&] {
    if (buffer == nullptr)
      throw std::invalid_argument("fill: null buffer");
    specfem::plugin::Buffer window(*buffer);
    handle.plugin->fill(first_step, window);
  });
}

/**
 * @brief ABI last_error.
 * @param self Handle from create, or nullptr for the last failed create
 * @return NUL-terminated message (empty when there is no error)
 */
inline const char *last_error(const void *self) noexcept {
  if (self == nullptr)
    return create_error().c_str();
  return static_cast<const Handle *>(self)->last_error.c_str();
}

/**
 * @brief ABI destroy.
 * @param self Handle from create (nullptr is a no-op)
 */
inline void destroy(void *self) noexcept { delete static_cast<Handle *>(self); }

} // namespace plugin_impl
} // namespace specfem

/**
 * @brief Register a C++ plugin class as this library's injection plugin.
 *
 * Use once per plugin, at global scope. Emits the manifest, the ABI function
 * table, and the exported @c specfempp_injection_plugin entry point.
 *
 * @param NAME String literal, at most 63 characters
 * @param VERSION String literal, at most 31 characters
 * @param PLUGIN_TYPE Class derived from specfem::plugin::InjectionPlugin
 */
#define SPECFEMPP_DEFINE_INJECTION_PLUGIN(NAME, VERSION, PLUGIN_TYPE)          \
  static_assert(                                                               \
      std::is_base_of_v<specfem::plugin::InjectionPlugin, PLUGIN_TYPE>,        \
      #PLUGIN_TYPE " must derive from specfem::plugin::InjectionPlugin");      \
  static_assert(                                                               \
      std::is_constructible_v<PLUGIN_TYPE, const std::string &,                \
                              const specfem::plugin::SimulationInfo &,         \
                              const specfem::plugin::Communicator &>,          \
      #PLUGIN_TYPE " must be constructible from (const std::string "           \
                   "&config_path, const specfem::plugin::SimulationInfo &, "   \
                   "const specfem::plugin::Communicator &)");                  \
  SPFP_DEFINE_INJECTION_MANIFEST(NAME, VERSION);                               \
  static const spfp_injection_plugin spfp_injection_plugin_table = {           \
    SPFP_INJECTION_ABI_VERSION,                                                \
    0u,                                                                        \
    0u,                                                                        \
    NAME,                                                                      \
    VERSION,                                                                   \
    &specfem::plugin_impl::create<PLUGIN_TYPE>,                                \
    &specfem::plugin_impl::supports,                                           \
    &specfem::plugin_impl::setup,                                              \
    &specfem::plugin_impl::fill,                                               \
    &specfem::plugin_impl::last_error,                                         \
    &specfem::plugin_impl::destroy                                             \
  };                                                                           \
  extern "C" SPFP_EXPORT const spfp_injection_plugin *                         \
  specfempp_injection_plugin(void) {                                           \
    return &spfp_injection_plugin_table;                                       \
  }
