#include "specfem/injection/session/provider_session.hpp"
#include "specfem/mpi.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>

namespace specfem::injection_impl {

/** @brief ABI dtype matching type_real. */
inline constexpr specfem_dtype type_real_dtype =
    std::is_same_v<type_real, float> ? specfem_dtype_float32
                                     : specfem_dtype_float64;

/**
 * @brief Make all ranks agree on the outcome of a collective plugin call.
 *
 * @param plugin Plugin name, for the message
 * @param call Name of the call, for the message
 * @param local_error Error on this rank, if any
 * @throws std::runtime_error on every rank if any rank failed
 */
void agree_on_outcome(const std::string &plugin, const std::string_view call,
                      const std::optional<std::string> &local_error);

/**
 * @brief Run one step of a collective plugin call and agree on its outcome.
 *
 * @tparam Step Callable returning std::optional<std::string>: the error on
 * this rank, or nullopt on success. Exceptions count as errors.
 * @param plugin Plugin name, for the message
 * @param call Name of the call, for the message
 * @param step The work to run on this rank
 * @throws std::runtime_error on every rank if @p step failed on any rank
 */
template <typename Step>
void collective(const std::string &plugin, const std::string_view call,
                Step &&step);

/** @brief Name of a medium, or empty for an unknown one. */
constexpr std::string_view medium_name(const specfem_medium medium) {
  switch (medium) {
  case specfem_medium_acoustic:
    return "acoustic";
  case specfem_medium_elastic:
    return "elastic";
  case specfem_medium_elastic_spin:
    return "elastic_spin";
  default:
    return {};
  }
}

/** @brief Name of a property, or empty for an unknown one. */
constexpr std::string_view property_name(const specfem_property property) {
  switch (property) {
  case specfem_property_isotropic:
    return "isotropic";
  case specfem_property_anisotropic:
    return "anisotropic";
  case specfem_property_isotropic_cosserat:
    return "isotropic_cosserat";
  default:
    return {};
  }
}

/** @brief Name of a quantity, or empty for an unknown one. */
constexpr std::string_view quantity_name(const specfem_quantity quantity) {
  switch (quantity) {
  case specfem_quantity_displacement:
    return "displacement";
  case specfem_quantity_velocity:
    return "velocity";
  case specfem_quantity_acceleration:
    return "acceleration";
  case specfem_quantity_traction:
    return "traction";
  case specfem_quantity_potential_rate:
    return "potential_rate";
  default:
    return {};
  }
}

/**
 * @brief Check that the groups are well formed and partition the points.
 * @param points Injection points on this rank
 * @return The first problem found, or nullopt
 */
std::optional<std::string>
check_groups(const specfem::injection::InjectionPoints &points);

/**
 * @brief Find the first value that differs on another rank.
 *
 * Collective: every rank must pass the same number of values.
 *
 * @param values Values on this rank
 * @return Index of the first value that differs across ranks, or nullopt
 */
std::optional<std::size_t>
first_difference_across_ranks(std::vector<int> values);

/**
 * @brief Check that every rank holds the same groups, in the same order.
 *
 * Collective: every rank makes the same MPI calls whatever its groups.
 *
 * @param groups Groups on this rank
 * @return A description of the mismatch, or nullopt
 */
std::optional<std::string> compare_groups_across_ranks(
    const std::vector<specfem::injection::PointGroup> &groups);

} // namespace specfem::injection_impl

void specfem::injection_impl::agree_on_outcome(
    const std::string &plugin, const std::string_view call,
    const std::optional<std::string> &local_error) {
  int failed = local_error ? 1 : 0;
  SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX,
                                     specfem::MPI::communicator()));
  if (failed == 0)
    return;
  const std::string prefix =
      "Injection plugin '" + plugin + "' " + std::string(call) + " failed ";
  if (!local_error)
    throw std::runtime_error(prefix + "on another rank.");
  throw std::runtime_error(prefix + "on rank " +
                           std::to_string(specfem::MPI::get_rank()) + ": " +
                           *local_error);
}

template <typename Step>
void specfem::injection_impl::collective(const std::string &plugin,
                                         const std::string_view call,
                                         Step &&step) {
  std::optional<std::string> error;
  try {
    error = step();
  } catch (const std::exception &e) {
    error = e.what();
  }
  specfem::injection_impl::agree_on_outcome(plugin, call, error);
}

std::optional<std::string> specfem::injection_impl::check_groups(
    const specfem::injection::InjectionPoints &points) {
  const auto nrows = points.size();
  std::int64_t next = 0;
  for (std::size_t g = 0; g < points.groups.size(); ++g) {
    const auto &group = points.groups[g];
    const std::string name = "point group " + std::to_string(g);
    if (specfem::injection_impl::medium_name(group.medium).empty())
      return name + " has an unknown medium " + std::to_string(group.medium) +
             ".";
    if (specfem::injection_impl::property_name(group.property).empty())
      return name + " has an unknown property " +
             std::to_string(group.property) + ".";
    if (group.begin != next || group.count < 0 ||
        group.count > nrows - group.begin)
      return name + " covers rows [" + std::to_string(group.begin) + ", " +
             std::to_string(group.begin + group.count) +
             "), but groups must partition the " + std::to_string(nrows) +
             " points in order.";
    if (static_cast<std::int64_t>(group.properties.extent(0)) != group.count ||
        (group.count > 0 &&
         group.properties.extent(1) != group.property_names.size()))
      return name + " has a property array that does not match its " +
             std::to_string(group.count) + " points and " +
             std::to_string(group.property_names.size()) + " property names.";
    for (const auto &request : group.quantities) {
      if (specfem::injection_impl::quantity_name(request.quantity).empty())
        return name + " requests an unknown quantity " +
               std::to_string(request.quantity) + ".";
      if (request.components < 1)
        return name + " requests " +
               std::string(
                   specfem::injection_impl::quantity_name(request.quantity)) +
               " with " + std::to_string(request.components) + " components.";
    }
    next = group.begin + group.count;
  }
  if (next != nrows)
    return "the point groups cover " + std::to_string(next) + " of the " +
           std::to_string(nrows) + " points.";
  return std::nullopt;
}

std::optional<std::size_t>
specfem::injection_impl::first_difference_across_ranks(
    std::vector<int> values) {
  std::vector<int> lowest = values;
  const int size = static_cast<int>(values.size());
  SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE, values.data(), size, MPI_INT,
                                     MPI_MAX, specfem::MPI::communicator()));
  SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE, lowest.data(), size, MPI_INT,
                                     MPI_MIN, specfem::MPI::communicator()));
  const auto mismatch = std::ranges::mismatch(values, lowest);
  if (mismatch.in1 == values.end())
    return std::nullopt;
  return static_cast<std::size_t>(mismatch.in1 - values.begin());
}

std::optional<std::string> specfem::injection_impl::compare_groups_across_ranks(
    const std::vector<specfem::injection::PointGroup> &groups) {
  const int ngroups = static_cast<int>(groups.size());
  if (specfem::injection_impl::first_difference_across_ranks({ ngroups }))
    return "ranks hold different numbers of point groups; every rank must "
           "hold the same groups.";

  // Each reduction needs the same length on every rank, so the quantity
  // lists are compared only once their lengths agree.
  std::vector<int> headers;
  headers.reserve(3 * groups.size());
  for (const auto &group : groups) {
    headers.push_back(group.medium);
    headers.push_back(group.property);
    headers.push_back(static_cast<int>(group.quantities.size()));
  }
  if (const auto index =
          specfem::injection_impl::first_difference_across_ranks(headers))
    return "point group " + std::to_string(*index / 3) +
           " has a different medium, property or number of quantities on "
           "another rank; every rank must hold the same groups.";

  std::vector<int> requests;
  std::vector<std::size_t> owners;
  for (std::size_t g = 0; g < groups.size(); ++g)
    for (const auto &request : groups[g].quantities) {
      requests.push_back(request.quantity);
      requests.push_back(request.components);
      owners.push_back(g);
    }
  if (const auto index =
          specfem::injection_impl::first_difference_across_ranks(requests))
    return "point group " + std::to_string(owners[*index / 2]) +
           " requests different quantities or components on another rank; "
           "every rank must hold the same groups.";
  return std::nullopt;
}

specfem::injection::ProviderSession::ProviderSession(
    specfem::injection::PluginLibrary library, const std::string &config_path,
    const SimulationParameters &parameters, const SessionOptions &options)
    : library_(std::move(library)), parameters_(parameters), options_(options) {
  specfem_simulation_info info{ .struct_size = sizeof(specfem_simulation_info),
                                .dimension = 3,
                                .nstep = parameters_.nstep,
                                .reserved = 0,
                                .t0 = parameters_.t0,
                                .dt = parameters_.dt };
  std::ranges::copy(parameters_.bounds, info.bounds);

#ifdef SPECFEM_ENABLE_MPI
  const std::int32_t has_mpi = 1;
  const auto fortran_handle =
      static_cast<std::int32_t>(MPI_Comm_c2f(specfem::MPI::communicator()));
#else
  const std::int32_t has_mpi = 0;
  const std::int32_t fortran_handle = 0;
#endif
  const specfem_communicator communicator{ .struct_size =
                                               sizeof(specfem_communicator),
                                           .has_mpi = has_mpi,
                                           .rank = specfem::MPI::get_rank(),
                                           .size = specfem::MPI::get_size(),
                                           .fortran_handle = fortran_handle };

  specfem::injection_impl::collective(
      library_.name(), "create", [&]() -> std::optional<std::string> {
        instance_.emplace(library_.create(config_path, info, communicator));
        return std::nullopt;
      });
}

void specfem::injection::ProviderSession::setup(const InjectionPoints &points) {
  const auto &table = instance_->table();
  void *self = instance_->get();

  // Each stage agrees across ranks before the next, so either every rank
  // calls the plugin's setup or none does.
  specfem::injection_impl::collective(
      library_.name(), "setup", [&]() -> std::optional<std::string> {
        return specfem::injection_impl::check_groups(points);
      });
  specfem::injection_impl::collective(
      library_.name(), "setup", [&]() -> std::optional<std::string> {
        return specfem::injection_impl::compare_groups_across_ranks(
            points.groups);
      });

  specfem::injection_impl::collective(
      library_.name(), "setup", [&]() -> std::optional<std::string> {
        std::string unsupported;
        for (const auto &group : points.groups)
          for (const auto &request : group.quantities)
            if (table.supports(self, group.medium, group.property,
                               request.quantity) == 0) {
              if (!unsupported.empty())
                unsupported += ", ";
              unsupported += specfem::injection_impl::medium_name(group.medium);
              unsupported += "/";
              unsupported +=
                  specfem::injection_impl::property_name(group.property);
              unsupported += "/";
              unsupported +=
                  specfem::injection_impl::quantity_name(request.quantity);
            }
        if (!unsupported.empty())
          return "it does not support " + unsupported + ".";
        return std::nullopt;
      });

  nrows_ = points.size();
  specfem::injection_impl::collective(
      library_.name(), "setup", [&]() -> std::optional<std::string> {
        // The plugin ABI takes double precision; the copies live for this
        // call.
        const auto to_double = [&](const InjectionPoints::view_type &view) {
          return std::vector<double>(view.data(), view.data() + nrows_);
        };
        const auto x = to_double(points.x), y = to_double(points.y),
                   z = to_double(points.z), nx = to_double(points.nx),
                   ny = to_double(points.ny), nz = to_double(points.nz);

        const auto ngroups = points.groups.size();
        std::vector<std::vector<const char *>> names(ngroups);
        std::vector<std::vector<double>> properties(ngroups);
        std::vector<std::vector<specfem_quantity_slot>> slots(ngroups);
        std::vector<specfem_point_group> groups(ngroups);
        std::vector<const specfem_point_group *> group_pointers(ngroups);
        for (std::size_t g = 0; g < ngroups; ++g) {
          const auto &group = points.groups[g];
          for (const auto &name : group.property_names)
            names[g].push_back(name.c_str());
          properties[g].assign(group.properties.data(),
                               group.properties.data() +
                                   group.properties.size());
          int offset = 0;
          for (const auto &request : group.quantities) {
            slots[g].push_back({ .quantity = request.quantity,
                                 .offset = offset,
                                 .count = request.components,
                                 .reserved = 0 });
            offset += request.components;
          }
          groups[g] = { .struct_size = sizeof(specfem_point_group),
                        .medium = group.medium,
                        .property = group.property,
                        .nproperties =
                            static_cast<std::int32_t>(names[g].size()),
                        .begin = group.begin,
                        .count = group.count,
                        .property_names = names[g].data(),
                        .properties = properties[g].empty()
                                          ? nullptr
                                          : properties[g].data(),
                        .ncomponents = offset,
                        .nslots = static_cast<std::int32_t>(slots[g].size()),
                        .slots = slots[g].data() };
          group_pointers[g] = &groups[g];
        }

        const specfem_points abi_points{ .struct_size = sizeof(specfem_points),
                                         .ngroups =
                                             static_cast<std::int32_t>(ngroups),
                                         .n = nrows_,
                                         .x = x.data(),
                                         .y = y.data(),
                                         .z = z.data(),
                                         .nx = nx.data(),
                                         .ny = ny.data(),
                                         .nz = nz.data(),
                                         .groups = group_pointers.data() };
        if (table.setup(self, &abi_points) != specfem_status_ok)
          return instance_->last_error();
        return std::nullopt;
      });

  groups_.clear();
  nslots_ = 0;
  for (const auto &group : points.groups) {
    groups_.push_back({ group.begin, group.count, group.components() });
    nslots_ = std::max(nslots_, group.components());
  }
  choose_window_steps();
  staging_ = staging_view("injection::staging", nrows_, nslots_, window_steps_);
  frame_ = FrameBuffer{ .data = FrameBuffer::view_type(
                            "injection::frame", nrows_, nslots_, window_steps_),
                        .window_start = 0,
                        .nsteps = 0 };
}

void specfem::injection::ProviderSession::choose_window_steps() {
  if (options_.window_steps && *options_.window_steps < 1)
    throw std::runtime_error("Injection window-steps must be at least 1, got " +
                             std::to_string(*options_.window_steps) + ".");
  int steps =
      options_.window_steps
          ? std::min(*options_.window_steps, std::max(parameters_.nstep, 1))
          : specfem::injection::window_steps_for_budget(
                options_.memory_budget, nrows_, nslots_, parameters_.nstep);

  // Ranks hold different numbers of points, but fill is collective, so every
  // rank must use the same window size.
  SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE, &steps, 1, MPI_INT, MPI_MIN,
                                     specfem::MPI::communicator()));
  window_steps_ = steps;
}

void specfem::injection::ProviderSession::ensure_window(const int istep) {
  if (window_steps_ == 0)
    throw std::logic_error(
        "ProviderSession::ensure_window called before setup.");
  if (istep < 0 || istep >= parameters_.nstep)
    throw std::out_of_range("Injection step " + std::to_string(istep) +
                            " is outside [0, " +
                            std::to_string(parameters_.nstep) + ").");
  // frame_.nsteps is 0 until the first window is filled.
  if (istep >= frame_.window_start &&
      istep < frame_.window_start + frame_.nsteps)
    return;

  const int start = specfem::injection::window_start(istep, window_steps_);
  const int nsteps = std::min(window_steps_, parameters_.nstep - start);

  specfem::injection_impl::collective(
      library_.name(), "fill", [&]() -> std::optional<std::string> {
        if (auto error = fill_window(start, nsteps))
          return error;
        return options_.validate ? validate_window(start, nsteps)
                                 : std::nullopt;
      });

  Kokkos::deep_copy(frame_.data, staging_);
  frame_.window_start = start;
  frame_.nsteps = nsteps;
}

std::optional<std::string>
specfem::injection::ProviderSession::fill_window(const int start,
                                                 const int nsteps) {
  // Each group's buffer is a strided view of its own rows of the staging
  // array, holding only that group's components.
  std::vector<specfem_buffer> buffers;
  buffers.reserve(groups_.size());
  for (const auto &group : groups_) {
    const bool empty = group.count == 0 || group.components == 0;
    buffers.push_back(
        { .struct_size = sizeof(specfem_buffer),
          .dtype = specfem::injection_impl::type_real_dtype,
          .data = empty ? nullptr
                        : staging_.data() + group.begin * staging_.stride(0),
          .npoints = group.count,
          .ncomponents = group.components,
          .nsteps = nsteps,
          .stride_point = static_cast<std::int64_t>(staging_.stride(0)),
          .stride_component = static_cast<std::int64_t>(staging_.stride(1)),
          .stride_step = static_cast<std::int64_t>(staging_.stride(2)) });
  }
  std::vector<const specfem_buffer *> pointers;
  pointers.reserve(buffers.size());
  for (const auto &buffer : buffers)
    pointers.push_back(&buffer);

  if (instance_->table().fill(instance_->get(), start,
                              static_cast<std::int32_t>(pointers.size()),
                              pointers.data()) != specfem_status_ok)
    return instance_->last_error();
  return std::nullopt;
}

std::optional<std::string>
specfem::injection::ProviderSession::validate_window(const int start,
                                                     const int nsteps) const {
  for (const auto &group : groups_)
    for (std::int64_t row = group.begin; row < group.begin + group.count; ++row)
      for (int slot = 0; slot < group.components; ++slot)
        for (int step = 0; step < nsteps; ++step) {
          const type_real value = staging_(row, slot, step);
          if (!std::isfinite(value)) {
            std::ostringstream message;
            message << "non-finite value " << value << " at point " << row
                    << ", slot " << slot << ", step " << start + step << ".";
            return message.str();
          }
        }
  return std::nullopt;
}
