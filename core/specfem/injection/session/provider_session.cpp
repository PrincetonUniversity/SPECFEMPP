#include "specfem/injection/session/provider_session.hpp"
#include "specfem/mpi.hpp"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>

namespace specfem::injection_impl {

/** @brief ABI dtype matching type_real. */
inline constexpr spfp_dtype type_real_dtype =
    std::is_same_v<type_real, float> ? spfp_dtype_float32 : spfp_dtype_float64;

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

/** @brief Human-readable name of a medium. */
constexpr std::string_view medium_name(const spfp_medium medium) {
  switch (medium) {
  case spfp_medium_elastic:
    return "elastic";
  case spfp_medium_acoustic:
    return "acoustic";
  default:
    return "unknown medium";
  }
}

/** @brief Human-readable name of a quantity. */
constexpr std::string_view quantity_name(const spfp_quantity quantity) {
  switch (quantity) {
  case spfp_quantity_displacement:
    return "displacement";
  case spfp_quantity_velocity:
    return "velocity";
  case spfp_quantity_acceleration:
    return "acceleration";
  case spfp_quantity_traction:
    return "traction";
  case spfp_quantity_potential_rate:
    return "potential_rate";
  default:
    return "unknown quantity";
  }
}

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

specfem::injection::ProviderSession::ProviderSession(
    specfem::injection::PluginLibrary library, const std::string &config_path,
    const SimulationParameters &parameters, const SessionOptions &options)
    : library_(std::move(library)), parameters_(parameters), options_(options) {
  spfp_simulation_info info{ .dimension = 3,
                             .nstep = parameters_.nstep,
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
  const spfp_communicator communicator{ .has_mpi = has_mpi,
                                        .rank = specfem::MPI::get_rank(),
                                        .size = specfem::MPI::get_size(),
                                        .fortran_handle = fortran_handle };

  specfem::injection_impl::collective(
      library_.name(), "create", [&]() -> std::optional<std::string> {
        instance_.emplace(library_.create(config_path, info, communicator));
        return std::nullopt;
      });
}

void specfem::injection::ProviderSession::setup(const InjectionPoints &points,
                                                const Request &request) {
  const auto &table = instance_->table();
  void *self = instance_->get();
  nrows_ = points.size();
  medium_.assign(points.medium.data(), points.medium.data() + nrows_);

  // Check everything locally first and agree on it, so either every rank
  // calls the plugin's setup or none does.
  specfem::injection_impl::collective(
      library_.name(), "setup", [&]() -> std::optional<std::string> {
        std::string unsupported;
        for (const auto medium : specfem::injection::media)
          for (const auto quantity : request.quantities(medium))
            if (table.supports(self, medium, quantity) == 0) {
              if (!unsupported.empty())
                unsupported += ", ";
              unsupported += specfem::injection_impl::medium_name(medium);
              unsupported += "/";
              unsupported += specfem::injection_impl::quantity_name(quantity);
            }
        if (!unsupported.empty())
          return "it does not support " + unsupported + ".";

        for (std::int64_t row = 0; row < nrows_; ++row)
          if (std::ranges::find(specfem::injection::media, medium_[row]) ==
              specfem::injection::media.end())
            return "injection point " + std::to_string(row) +
                   " has an unknown medium " + std::to_string(medium_[row]) +
                   ".";
        return std::nullopt;
      });

  specfem::injection_impl::collective(
      library_.name(), "setup", [&]() -> std::optional<std::string> {
        // The plugin ABI takes double precision; the copies live for this
        // call.
        const auto to_double = [&](const InjectionPoints::view_type &view) {
          return std::vector<double>(view.data(), view.data() + nrows_);
        };
        const auto x = to_double(points.x), y = to_double(points.y),
                   z = to_double(points.z), nx = to_double(points.nx),
                   ny = to_double(points.ny), nz = to_double(points.nz),
                   rho = to_double(points.rho), vp = to_double(points.vp),
                   vs = to_double(points.vs);
        const spfp_points abi_points{ .n = nrows_,
                                      .x = x.data(),
                                      .y = y.data(),
                                      .z = z.data(),
                                      .nx = nx.data(),
                                      .ny = ny.data(),
                                      .nz = nz.data(),
                                      .medium = medium_.data(),
                                      .rho = rho.data(),
                                      .vp = vp.data(),
                                      .vs = vs.data() };

        request_ = request;
        const spfp_request abi_request{
          .n_elastic = static_cast<std::int32_t>(request_.elastic.size()),
          .elastic = request_.elastic.data(),
          .n_acoustic = static_cast<std::int32_t>(request_.acoustic.size()),
          .acoustic = request_.acoustic.data()
        };
        if (table.setup(self, &abi_points, &abi_request) != spfp_status_ok)
          return instance_->last_error();
        return std::nullopt;
      });

  nslots_ = request_.max_components();
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
  const spfp_buffer buffer{
    .data = staging_.data(),
    .dtype = specfem::injection_impl::type_real_dtype,
    .ncomponents = nslots_,
    .npoints = nrows_,
    .nsteps = nsteps,
    .stride_point = static_cast<std::int64_t>(staging_.stride(0)),
    .stride_component = static_cast<std::int64_t>(staging_.stride(1)),
    .stride_step = static_cast<std::int64_t>(staging_.stride(2))
  };
  if (instance_->table().fill(instance_->get(), start, &buffer) !=
      spfp_status_ok)
    return instance_->last_error();
  return std::nullopt;
}

std::optional<std::string>
specfem::injection::ProviderSession::validate_window(const int start,
                                                     const int nsteps) const {
  for (std::int64_t row = 0; row < nrows_; ++row) {
    const int slots =
        request_.components(static_cast<spfp_medium>(medium_[row]));
    for (int slot = 0; slot < slots; ++slot)
      for (int step = 0; step < nsteps; ++step) {
        const type_real value = staging_(row, slot, step);
        if (!std::isfinite(value)) {
          std::ostringstream message;
          message << "non-finite value " << value << " at point " << row
                  << ", slot " << slot << ", step " << start + step << ".";
          return message.str();
        }
      }
  }
  return std::nullopt;
}
