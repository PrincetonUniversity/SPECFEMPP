#pragma once

#include "specfem/plugin/injection_abi.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace specfem {
/**
 * @brief C++ layer of the injection plugin SDK.
 *
 * Plugin authors derive from @c InjectionPlugin and register the class with
 * @c SPECFEMPP_DEFINE_INJECTION_PLUGIN; the macro implements the C ABI in
 * @c %specfem/plugin/injection_abi.h on top of it.
 */
namespace plugin {

/** @brief Medium of a point group (SPECFEM++ medium tag identifiers). */
enum class Medium : std::int32_t {
  acoustic = specfempp_medium_acoustic,        ///< Acoustic fluid
  elastic = specfempp_medium_elastic,          ///< Elastic solid
  elastic_spin = specfempp_medium_elastic_spin ///< Elastic solid with spin
};

/** @brief Material parameterization (SPECFEM++ property tag identifiers). */
enum class Property : std::int32_t {
  isotropic = specfempp_property_isotropic,     ///< Isotropic
  anisotropic = specfempp_property_anisotropic, ///< Anisotropic
  isotropic_cosserat =
      specfempp_property_isotropic_cosserat ///< Isotropic Cosserat
};

/** @brief Physical quantity SPECFEM++ can request from a plugin. */
enum class Quantity : std::int32_t {
  displacement = specfempp_quantity_displacement,    ///< Displacement
  velocity = specfempp_quantity_velocity,            ///< Velocity
  acceleration = specfempp_quantity_acceleration,    ///< Acceleration
  traction = specfempp_quantity_traction,            ///< Traction on the normal
  potential_rate = specfempp_quantity_potential_rate ///< Fluid potential rate
};

/**
 * @brief Whether a raw ABI value names a known medium.
 *
 * @param medium Raw medium value
 * @return true for every medium defined in ABI v1
 */
constexpr bool is_valid_medium(const specfempp_medium medium) {
  return medium == specfempp_medium_acoustic ||
         medium == specfempp_medium_elastic ||
         medium == specfempp_medium_elastic_spin;
}

/**
 * @brief Whether a raw ABI value names a known property.
 *
 * @param property Raw property value
 * @return true for every property defined in ABI v1
 */
constexpr bool is_valid_property(const specfempp_property property) {
  return property >= specfempp_property_isotropic &&
         property <= specfempp_property_isotropic_cosserat;
}

/**
 * @brief Whether a raw ABI value names a known quantity.
 *
 * @param quantity Raw quantity value
 * @return true for every quantity defined in ABI v1
 */
constexpr bool is_valid_quantity(const specfempp_quantity quantity) {
  return quantity >= specfempp_quantity_displacement &&
         quantity <= specfempp_quantity_potential_rate;
}

/**
 * @brief Thrown by a plugin that cannot serve the run.
 *
 * The SDK reports it as @c specfempp_status_unsupported; any other exception
 * becomes @c specfempp_status_error.
 */
class Unsupported : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

} // namespace plugin

namespace plugin_impl {

/**
 * @brief Check that an extensible ABI struct holds at least the v1 fields.
 * @tparam T ABI struct type with a leading struct_size
 * @param value Struct from the other side of the ABI
 * @param v1_size Size of the struct in ABI v1
 * @param what Name used in the error message
 * @throws std::invalid_argument if the struct is too small
 */
template <typename T>
void require_size(const T &value, const std::size_t v1_size, const char *what) {
  if (value.struct_size < v1_size)
    throw std::invalid_argument(std::string(what) + ": struct_size " +
                                std::to_string(value.struct_size) +
                                " is smaller than the ABI v1 size " +
                                std::to_string(v1_size));
}

/** @brief Size of specfempp_simulation_info in ABI v1. */
inline constexpr std::size_t simulation_info_v1_size =
    offsetof(specfempp_simulation_info, bounds) +
    sizeof(specfempp_simulation_info::bounds);
/** @brief Size of specfempp_communicator in ABI v1. */
inline constexpr std::size_t communicator_v1_size =
    offsetof(specfempp_communicator, fortran_handle) + sizeof(std::int32_t);
/** @brief Size of specfempp_point_group in ABI v1. */
inline constexpr std::size_t point_group_v1_size =
    offsetof(specfempp_point_group, slots) + sizeof(void *);
/** @brief Size of specfempp_points in ABI v1. */
inline constexpr std::size_t points_v1_size =
    offsetof(specfempp_points, groups) + sizeof(void *);
/** @brief Size of specfempp_buffer in ABI v1. */
inline constexpr std::size_t buffer_v1_size =
    offsetof(specfempp_buffer, stride_step) + sizeof(std::int64_t);

} // namespace plugin_impl

namespace plugin {

/** @brief Simulation parameters passed to the plugin at creation. */
class SimulationInfo {
public:
  /**
   * @brief Wrap and validate the ABI struct.
   * @param info Simulation parameters from SPECFEM++
   * @throws std::invalid_argument if the struct is too small
   * @throws Unsupported if the dimension is not 3
   */
  explicit SimulationInfo(const specfempp_simulation_info &info) : info_(info) {
    specfem::plugin_impl::require_size(
        info, specfem::plugin_impl::simulation_info_v1_size, "simulation_info");
    if (info_.dimension != 3)
      throw Unsupported("simulation_info: ABI v1 supports dimension 3 only, "
                        "got " +
                        std::to_string(info_.dimension));
    if (info_.nstep < 0)
      throw std::invalid_argument("simulation_info: negative nstep");
  }

  /** @brief Spatial dimension (3 in ABI v1). @return dimension */
  int dimension() const { return info_.dimension; }

  /** @brief Total number of time steps. @return step count */
  int number_of_steps() const { return info_.nstep; }

  /** @brief Time of step 0. @return t0 */
  double t0() const { return info_.t0; }

  /** @brief Time step. @return dt */
  double dt() const { return info_.dt; }

  /**
   * @brief Time of a simulation step.
   * @param step Simulation step index
   * @return @f$ t_0 + \mathrm{step}\,\Delta t @f$
   */
  double time(const int step) const { return info_.t0 + step * info_.dt; }

  /**
   * @brief Bounding box of the global domain, identical on every rank.
   * @return xmin, xmax, ymin, ymax, zmin, zmax
   */
  std::array<double, 6> bounds() const {
    return { info_.bounds[0], info_.bounds[1], info_.bounds[2],
             info_.bounds[3], info_.bounds[4], info_.bounds[5] };
  }

private:
  specfempp_simulation_info info_; ///< Copy of the ABI struct
};

/** @brief MPI communicator passed to the plugin at creation. */
class Communicator {
public:
  /**
   * @brief Wrap and validate the ABI struct.
   * @param communicator Communicator description from SPECFEM++
   * @throws std::invalid_argument if the struct is too small
   */
  explicit Communicator(const specfempp_communicator &communicator)
      : communicator_(communicator) {
    specfem::plugin_impl::require_size(
        communicator, specfem::plugin_impl::communicator_v1_size,
        "communicator");
  }

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
  specfempp_communicator communicator_; ///< Copy of the ABI struct
};

/** @brief Layout of one requested quantity in a group's buffer. */
struct QuantitySlot {
  Quantity quantity; ///< Requested quantity
  int offset;        ///< First buffer component
  int count;         ///< Number of components
};

/**
 * @brief Points of one material system and the quantities requested there.
 *
 * The metadata (medium, property, ranges, property names, layout) is copied
 * and stays valid; the property values are read from SPECFEM++ memory and are
 * only valid during @c setup.
 */
class PointGroup {
public:
  /**
   * @brief Copy and validate the ABI group.
   * @param group Group from SPECFEM++
   * @param npoints Total number of points, to check the group's range
   * @throws std::invalid_argument on any inconsistent field
   */
  PointGroup(const specfempp_point_group &group, const std::int64_t npoints) {
    specfem::plugin_impl::require_size(
        group, specfem::plugin_impl::point_group_v1_size, "point_group");
    if (!is_valid_medium(group.medium))
      throw std::invalid_argument("point_group: unknown medium " +
                                  std::to_string(group.medium));
    if (!is_valid_property(group.property))
      throw std::invalid_argument("point_group: unknown property " +
                                  std::to_string(group.property));
    if (group.begin < 0 || group.count < 0 ||
        group.begin > npoints - group.count)
      throw std::invalid_argument("point_group: range outside the points");
    if (group.nproperties < 0 ||
        (group.nproperties > 0 && group.property_names == nullptr) ||
        (group.nproperties > 0 && group.count > 0 &&
         group.properties == nullptr))
      throw std::invalid_argument("point_group: invalid property arrays");
    if (group.ncomponents < 0 || group.nslots < 0 ||
        (group.nslots > 0 && group.slots == nullptr))
      throw std::invalid_argument("point_group: invalid quantity slots");

    medium_ = static_cast<Medium>(group.medium);
    property_ = static_cast<Property>(group.property);
    begin_ = group.begin;
    count_ = group.count;
    ncomponents_ = group.ncomponents;
    properties_ = group.properties;

    names_.reserve(group.nproperties);
    for (std::int32_t k = 0; k < group.nproperties; ++k) {
      if (group.property_names[k] == nullptr)
        throw std::invalid_argument("point_group: null property name");
      names_.emplace_back(group.property_names[k]);
    }

    slots_.reserve(group.nslots);
    for (std::int32_t i = 0; i < group.nslots; ++i) {
      const auto &slot = group.slots[i];
      if (!is_valid_quantity(slot.quantity))
        throw std::invalid_argument("point_group: unknown quantity " +
                                    std::to_string(slot.quantity));
      if (slot.offset < 0 || slot.count <= 0 ||
          slot.offset > group.ncomponents - slot.count)
        throw std::invalid_argument(
            "point_group: quantity slot outside the components");
      slots_.push_back(
          { static_cast<Quantity>(slot.quantity), slot.offset, slot.count });
    }
  }

  /** @brief Medium of the group. @return medium */
  Medium medium() const { return medium_; }

  /** @brief Material parameterization. @return property */
  Property property() const { return property_; }

  /** @brief First point of the group in Points. @return index */
  std::int64_t begin() const { return begin_; }

  /** @brief Number of points in the group. @return point count */
  std::int64_t size() const { return count_; }

  /** @brief Names of the material properties. @return ordered names */
  const std::vector<std::string> &property_names() const { return names_; }

  /**
   * @brief Position of a material property.
   * @param name Property name
   * @return index, or -1 if the group has no such property
   */
  int property_index(const std::string_view name) const {
    for (std::size_t k = 0; k < names_.size(); ++k)
      if (names_[k] == name)
        return static_cast<int>(k);
    return -1;
  }

  /**
   * @brief Material property of a point (valid only during setup).
   * @param point Point index within the group
   * @param index Property index
   * @return property value
   * @throws std::out_of_range on an invalid point or index
   */
  double property(const std::int64_t point, const int index) const {
    if (point < 0 || point >= count_ || index < 0 ||
        index >= static_cast<int>(names_.size()))
      throw std::out_of_range("point_group: property access out of range");
    return properties_[point * static_cast<std::int64_t>(names_.size()) +
                       index];
  }

  /**
   * @brief Material property of a point by name (valid only during setup).
   * @param point Point index within the group
   * @param name Property name
   * @return property value
   * @throws std::out_of_range if the group has no such property
   */
  double property(const std::int64_t point, const std::string_view name) const {
    const int index = property_index(name);
    if (index < 0)
      throw std::out_of_range("point_group: no property '" + std::string(name) +
                              "'");
    return property(point, index);
  }

  /** @brief Buffer components per point. @return component count */
  int number_of_components() const { return ncomponents_; }

  /** @brief Requested quantities and their layout. @return slots */
  const std::vector<QuantitySlot> &slots() const { return slots_; }

  /**
   * @brief Layout of a requested quantity.
   * @param quantity Quantity
   * @return slot, or nullptr if @p quantity is not requested
   */
  const QuantitySlot *slot(const Quantity quantity) const {
    for (const auto &slot : slots_)
      if (slot.quantity == quantity)
        return &slot;
    return nullptr;
  }

private:
  Medium medium_ = Medium::elastic;         ///< Medium
  Property property_ = Property::isotropic; ///< Material parameterization
  std::int64_t begin_ = 0;                  ///< First point
  std::int64_t count_ = 0;                  ///< Number of points
  int ncomponents_ = 0;                     ///< Buffer components per point
  const double *properties_ = nullptr;      ///< Values (not owned)
  std::vector<std::string> names_;          ///< Property names
  std::vector<QuantitySlot> slots_;         ///< Requested quantities
};

/**
 * @brief Read-only view of the injection points, valid only during setup.
 *
 * Copy whatever the plugin needs; the arrays are released after @c setup
 * returns. The groups partition the points in order.
 */
class Points {
public:
  /**
   * @brief Wrap and validate the ABI struct.
   * @param points Points from SPECFEM++
   * @throws std::invalid_argument on any inconsistent field
   */
  explicit Points(const specfempp_points &points) : points_(points) {
    specfem::plugin_impl::require_size(
        points, specfem::plugin_impl::points_v1_size, "points");
    if (points_.n < 0)
      throw std::invalid_argument("points: negative point count");
    if (points_.n > 0 && (points_.x == nullptr || points_.y == nullptr ||
                          points_.z == nullptr || points_.nx == nullptr ||
                          points_.ny == nullptr || points_.nz == nullptr))
      throw std::invalid_argument("points: null coordinate or normal array");
    if (points_.ngroups < 0 ||
        (points_.ngroups > 0 && points_.groups == nullptr))
      throw std::invalid_argument("points: invalid group list");

    groups_.reserve(points_.ngroups);
    std::int64_t end = 0;
    for (std::int32_t g = 0; g < points_.ngroups; ++g) {
      if (points_.groups[g] == nullptr)
        throw std::invalid_argument("points: null group");
      groups_.emplace_back(*points_.groups[g], points_.n);
      if (groups_.back().begin() != end)
        throw std::invalid_argument("points: groups do not partition the "
                                    "points in order");
      end += groups_.back().size();
    }
    if (end != points_.n)
      throw std::invalid_argument("points: groups do not cover every point");
  }

  /** @brief Number of points. @return point count */
  std::int64_t size() const { return points_.n; }

  /** @brief Groups of points, one per material system. @return groups */
  const std::vector<PointGroup> &groups() const { return groups_; }

  /** @brief x-coordinate. @param i point @return x */
  double x(const std::int64_t i) const { return points_.x[i]; }
  /** @brief y-coordinate. @param i point @return y */
  double y(const std::int64_t i) const { return points_.y[i]; }
  /** @brief z-coordinate (up). @param i point @return z */
  double z(const std::int64_t i) const { return points_.z[i]; }
  /** @brief Unit normal, x. @param i point @return nx */
  double nx(const std::int64_t i) const { return points_.nx[i]; }
  /** @brief Unit normal, y. @param i point @return ny */
  double ny(const std::int64_t i) const { return points_.ny[i]; }
  /** @brief Unit normal, z. @param i point @return nz */
  double nz(const std::int64_t i) const { return points_.nz[i]; }

private:
  specfempp_points points_;        ///< Copy of the ABI struct (not owned)
  std::vector<PointGroup> groups_; ///< Validated groups
};

/**
 * @brief Window of the host buffer of one group, valid only during fill.
 *
 * Point @c p is point @c p of the group; step @c s of the window is
 * simulation step @c first_step + s of the @c fill call.
 */
class Buffer {
public:
  /**
   * @brief Wrap and validate the ABI buffer.
   * @param buffer Buffer from SPECFEM++
   * @throws std::invalid_argument on a too small struct, negative sizes, null
   * data or an unknown dtype
   */
  explicit Buffer(const specfempp_buffer &buffer) : buffer_(buffer) {
    specfem::plugin_impl::require_size(
        buffer, specfem::plugin_impl::buffer_v1_size, "buffer");
    if (buffer_.npoints < 0 || buffer_.ncomponents < 0 || buffer_.nsteps < 0)
      throw std::invalid_argument("buffer: negative size");
    if (buffer_.data == nullptr && buffer_.npoints > 0 &&
        buffer_.ncomponents > 0 && buffer_.nsteps > 0)
      throw std::invalid_argument("buffer: null data");
    if (buffer_.dtype != specfempp_dtype_float32 &&
        buffer_.dtype != specfempp_dtype_float64)
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
   * @param point Point index within the group
   * @param component Component index
   * @param step Step index within the window
   * @param value Value to store
   * @throws std::out_of_range if an index is outside the buffer
   */
  void set(const std::int64_t point, const int component, const int step,
           const double value) const {
    if (point < 0 || point >= buffer_.npoints || component < 0 ||
        component >= buffer_.ncomponents || step < 0 || step >= buffer_.nsteps)
      throw std::out_of_range("buffer: index out of range");
    const std::int64_t offset = point * buffer_.stride_point +
                                component * buffer_.stride_component +
                                step * buffer_.stride_step;
    if (buffer_.dtype == specfempp_dtype_float32)
      static_cast<float *>(buffer_.data)[offset] = static_cast<float>(value);
    else
      static_cast<double *>(buffer_.data)[offset] = value;
  }

private:
  specfempp_buffer buffer_; ///< Copy of the ABI struct (data not owned)
};

/**
 * @brief Abstract base class for C++ injection plugins.
 *
 * Derive from this class, add a constructor taking
 * @c (const std::string &config_path, const SimulationInfo &, const
 * Communicator &), and register it with
 * @c SPECFEMPP_DEFINE_INJECTION_PLUGIN. Exceptions thrown from the constructor
 * or any member are converted to ABI status codes and error messages; throw
 * @c Unsupported when the plugin cannot serve the run.
 *
 * @code
 * class MyPlugin : public specfem::plugin::InjectionPlugin {
 * public:
 *   MyPlugin(const std::string &config_path,
 *            const specfem::plugin::SimulationInfo &info,
 *            const specfem::plugin::Communicator &communicator);
 *   bool supports(specfem::plugin::Medium medium,
 *                 specfem::plugin::Property property,
 *                 specfem::plugin::Quantity quantity) const override;
 *   void setup(const specfem::plugin::Points &points) override;
 *   void fill(int first_step,
 *             const std::vector<specfem::plugin::Buffer> &buffers) override;
 * };
 * SPECFEMPP_DEFINE_INJECTION_PLUGIN("my-plugin", "1.0.0", MyPlugin)
 * @endcode
 */
class InjectionPlugin {
public:
  virtual ~InjectionPlugin() = default;

  /**
   * @brief Whether the plugin can provide a quantity for a material system.
   * @param medium Medium of the points
   * @param property Material parameterization of the points
   * @param quantity Requested quantity
   * @return true if supported
   */
  virtual bool supports(Medium medium, Property property,
                        Quantity quantity) const = 0;

  /**
   * @brief Receive the injection points, their groups and the requested
   * quantities (called once).
   *
   * Copy what is needed: @p points is only valid during this call.
   *
   * @param points Injection points
   */
  virtual void setup(const Points &points) = 0;

  /**
   * @brief Fill one window of time steps for every group.
   *
   * @c buffers[g] belongs to group @c g of @c setup. The SDK has already
   * checked each buffer's size against its group. Must give the same values
   * regardless of window order or size.
   *
   * @param first_step Simulation step of window step 0
   * @param buffers One buffer per group
   */
  virtual void fill(int first_step, const std::vector<Buffer> &buffers) = 0;
};

} // namespace plugin

namespace plugin_impl {

/** @brief Capacity of an error message, including the terminating NUL. */
inline constexpr std::size_t error_capacity = 512;

/** @brief Fixed-size error message; storing one never allocates. */
using ErrorMessage = std::array<char, error_capacity>;

/**
 * @brief Store a message, truncating it to the capacity.
 * @param message Destination
 * @param text NUL-terminated text (nullptr stores an empty message)
 */
inline void set_error(ErrorMessage &message, const char *text) noexcept {
  std::size_t i = 0;
  if (text != nullptr)
    for (; i + 1 < message.size() && text[i] != '\0'; ++i)
      message[i] = text[i];
  message[i] = '\0';
}

/** @brief Shape of one group, recorded at setup to check fill buffers. */
struct GroupShape {
  std::int64_t npoints = 0; ///< Points in the group
  int ncomponents = 0;      ///< Buffer components per point
};

/** @brief Instance handle passed through the C ABI as @c void*. */
struct Handle {
  std::unique_ptr<specfem::plugin::InjectionPlugin> plugin; ///< Instance
  int nstep = 0;                  ///< Total number of simulation steps
  bool is_setup = false;          ///< Whether setup succeeded
  std::vector<GroupShape> shapes; ///< Group shapes from setup
  ErrorMessage last_error{};      ///< Last error message
};

/**
 * @brief Message of the last failed create on this thread.
 * @return reference to the thread-local message
 */
inline ErrorMessage &create_error() noexcept {
  static thread_local ErrorMessage message{};
  return message;
}

/**
 * @brief Run @p body, converting exceptions into a status and message.
 *
 * Storing the message never allocates, so this cannot terminate.
 *
 * @tparam Body Callable returning nothing
 * @param message Error message to update
 * @param body Work to run
 * @return specfempp_status_ok; specfempp_status_unsupported for
 * specfem::plugin::Unsupported; otherwise specfempp_status_error
 */
template <typename Body>
specfempp_status guarded(ErrorMessage &message, const Body &body) noexcept {
  try {
    body();
    set_error(message, "");
    return specfempp_status_ok;
  } catch (const specfem::plugin::Unsupported &error) {
    set_error(message, error.what());
    return specfempp_status_unsupported;
  } catch (const std::exception &error) {
    set_error(message, error.what());
  } catch (...) {
    set_error(message, "unknown exception");
  }
  return specfempp_status_error;
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
void *create(const char *config_path, const specfempp_simulation_info *info,
             const specfempp_communicator *communicator) noexcept {
  Handle *result = nullptr;
  guarded(create_error(), [&] {
    if (config_path == nullptr || info == nullptr || communicator == nullptr)
      throw std::invalid_argument("create: null argument");
    const specfem::plugin::SimulationInfo simulation_info(*info);
    auto handle = std::make_unique<Handle>();
    handle->nstep = simulation_info.number_of_steps();
    handle->plugin = std::make_unique<PluginType>(
        std::string(config_path), simulation_info,
        specfem::plugin::Communicator(*communicator));
    result = handle.release();
  });
  return result;
}

/**
 * @brief ABI supports.
 * @param self Handle from create
 * @param medium Raw medium
 * @param property Raw property
 * @param quantity Raw quantity
 * @return 1 if supported, else 0 (also for invalid arguments or exceptions)
 */
inline std::int32_t supports(void *self, const specfempp_medium medium,
                             const specfempp_property property,
                             const specfempp_quantity quantity) noexcept {
  if (self == nullptr || !specfem::plugin::is_valid_medium(medium) ||
      !specfem::plugin::is_valid_property(property) ||
      !specfem::plugin::is_valid_quantity(quantity))
    return 0;
  auto &handle = *static_cast<Handle *>(self);
  bool result = false;
  const auto status = guarded(handle.last_error, [&] {
    result = handle.plugin->supports(
        static_cast<specfem::plugin::Medium>(medium),
        static_cast<specfem::plugin::Property>(property),
        static_cast<specfem::plugin::Quantity>(quantity));
  });
  return (status == specfempp_status_ok && result) ? 1 : 0;
}

/**
 * @brief ABI setup.
 * @param self Handle from create
 * @param points Injection points and their groups
 * @return status
 */
inline specfempp_status setup(void *self,
                              const specfempp_points *points) noexcept {
  if (self == nullptr)
    return specfempp_status_error;
  auto &handle = *static_cast<Handle *>(self);
  return guarded(handle.last_error, [&] {
    if (points == nullptr)
      throw std::invalid_argument("setup: null argument");
    const specfem::plugin::Points view(*points);
    std::vector<GroupShape> shapes;
    shapes.reserve(view.groups().size());
    for (const auto &group : view.groups())
      shapes.push_back({ group.size(), group.number_of_components() });
    handle.is_setup = false;
    handle.plugin->setup(view);
    handle.shapes = std::move(shapes);
    handle.is_setup = true;
  });
}

/**
 * @brief ABI fill: validate the buffers against setup, then fill them.
 * @param self Handle from create
 * @param first_step Simulation step of window step 0
 * @param nbuffers Number of buffers (one per group)
 * @param buffers Buffers to fill
 * @return status
 */
inline specfempp_status fill(void *self, const std::int32_t first_step,
                             const std::int32_t nbuffers,
                             const specfempp_buffer *const *buffers) noexcept {
  if (self == nullptr)
    return specfempp_status_error;
  auto &handle = *static_cast<Handle *>(self);
  return guarded(handle.last_error, [&] {
    if (!handle.is_setup)
      throw std::logic_error("fill: setup has not succeeded");
    if (nbuffers < 0 ||
        static_cast<std::size_t>(nbuffers) != handle.shapes.size())
      throw std::invalid_argument("fill: expected one buffer per group");
    if (nbuffers > 0 && buffers == nullptr)
      throw std::invalid_argument("fill: null buffer list");
    std::vector<specfem::plugin::Buffer> windows;
    windows.reserve(nbuffers);
    for (std::int32_t g = 0; g < nbuffers; ++g) {
      if (buffers[g] == nullptr)
        throw std::invalid_argument("fill: null buffer");
      windows.emplace_back(*buffers[g]);
      const auto &window = windows.back();
      const auto &shape = handle.shapes[g];
      if (window.number_of_points() != shape.npoints ||
          window.number_of_components() != shape.ncomponents)
        throw std::invalid_argument("fill: buffer " + std::to_string(g) +
                                    " does not match its group");
      if (window.number_of_steps() != windows.front().number_of_steps())
        throw std::invalid_argument("fill: buffers differ in step count");
    }
    const int nsteps = windows.empty() ? 0 : windows.front().number_of_steps();
    if (first_step < 0 || first_step > handle.nstep - nsteps)
      throw std::out_of_range("fill: window outside the simulation steps");
    handle.plugin->fill(first_step, windows);
  });
}

/**
 * @brief ABI last_error.
 * @param self Handle from create, or nullptr for the last failed create
 * @return NUL-terminated message (empty when there is no error)
 */
inline const char *last_error(const void *self) noexcept {
  if (self == nullptr)
    return create_error().data();
  return static_cast<const Handle *>(self)->last_error.data();
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
  SPECFEMPP_DEFINE_INJECTION_MANIFEST(NAME, VERSION);                          \
  static const specfempp_injection_plugin_table                                \
      specfempp_injection_plugin_table_instance = {                            \
        sizeof(specfempp_injection_plugin_table),                              \
        SPECFEMPP_INJECTION_ABI_VERSION,                                       \
        0u,                                                                    \
        NAME,                                                                  \
        VERSION,                                                               \
        &specfem::plugin_impl::create<PLUGIN_TYPE>,                            \
        &specfem::plugin_impl::supports,                                       \
        &specfem::plugin_impl::setup,                                          \
        &specfem::plugin_impl::fill,                                           \
        &specfem::plugin_impl::last_error,                                     \
        &specfem::plugin_impl::destroy                                         \
      };                                                                       \
  extern "C" SPECFEMPP_EXPORT const specfempp_injection_plugin_table *         \
  specfempp_injection_plugin(void) {                                           \
    return &specfempp_injection_plugin_table_instance;                         \
  }
