#include "specfem/globe/model_evaluator.hpp"

#include "globe_model_evaluator.h"
#include "specfem/globe/dimensionalization.hpp"
#include "specfem/mpi.hpp"
#include "specfem/units.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace specfem::globe::evaluator_impl {

constexpr int status_ok = 0;
constexpr int status_already_initialized = 1;
constexpr int status_not_initialized = 2;
constexpr int status_unsupported_model = 3;
constexpr int status_imain_open_failed = 4;
constexpr int status_bad_argument = 5;
constexpr std::size_t model_code_count = 5;
constexpr std::size_t model_flag_count = 16;

// Every catalog call is serialized because upstream model routines retain
// module and SAVE scratch state and are not reentrant. Recursive locking lets
// construction use the same guarded public accessors for its cross-checks.
std::recursive_mutex catalog_mutex;

std::string describe_status(const int status) {
  switch (status) {
  case status_ok:
    return "success";
  case status_already_initialized:
    return "the globe model catalog is already configured; only one "
           "specfem::globe::ModelEvaluator may own its Fortran module state";
  case status_not_initialized:
    return "the globe model catalog has not been configured";
  case status_unsupported_model:
    return "this model requires per-GLL indexing into the mesher's own "
           "discretization and cannot be evaluated from position alone";
  case status_imain_open_failed:
    return "could not open the log file for the globe model catalog";
  case status_bad_argument:
    return "the globe model catalog rejected an argument as out of range";
  default:
    return "unrecognized status code " + std::to_string(status);
  }
}

void require_ok(const int status, const std::string &operation) {
  if (status != status_ok) {
    throw std::runtime_error("specfem::globe::ModelEvaluator::" + operation +
                             ": " + describe_status(status));
  }
}

bool is_data_independent_model(std::string model_name) {
  std::transform(model_name.begin(), model_name.end(), model_name.begin(),
                 [](const unsigned char value) {
                   return static_cast<char>(std::tolower(value));
                 });
  const auto option = model_name.find_first_of("+, ");
  if (option != std::string::npos) {
    model_name.resize(option);
  }
  return model_name == "prem" || model_name == "1d_isotropic_prem" ||
         model_name == "1d_transversely_isotropic_prem";
}

void preflight_model_data(const ModelConfig &config) {
  if (is_data_independent_model(config.model_name)) {
    return;
  }
  const std::filesystem::path expected =
      std::filesystem::current_path() / "DATA";
  if (!std::filesystem::is_directory(expected)) {
    throw std::runtime_error(
        "specfem::globe::ModelEvaluator: model '" + config.model_name +
        "' may require runtime model files, but the expected DATA/ directory "
        "does not exist at '" +
        expected.string() +
        "'; run SPECFEM++ from a globe model directory containing DATA/");
  }
}

void check_model_config(const ModelConfig &config) {
  if (config.catalog_codes.empty() && config.catalog_flags.empty()) {
    return;
  }

  std::array<int, model_code_count> codes{};
  std::array<int, model_flag_count> flags{};
  const int status = globe_evaluator_model_config(codes.data(), flags.data());
  require_ok(status, "check_model_config");

  if ((!config.catalog_codes.empty() &&
       !std::equal(codes.begin(), codes.end(), config.catalog_codes.begin())) ||
      (!config.catalog_flags.empty() &&
       !std::equal(flags.begin(), flags.end(), config.catalog_flags.begin()))) {
    throw std::runtime_error(
        "Globe database MODEL_CONFIG disagrees with the model catalog linked "
        "into SPECFEM++; regenerate the database and executable from matching "
        "SPECFEM3D_GLOBE sources");
  }
}

void check_scales(const PlanetConstants &constants,
                  const double evaluator_r_planet,
                  const double evaluator_rhoav) {
  constexpr double relative_tolerance = 1.0e-12;
  const auto &values = constants.values();
  if (!std::isfinite(evaluator_r_planet) ||
      std::abs(evaluator_r_planet - values.r_planet) >
          relative_tolerance * values.r_planet) {
    std::ostringstream message;
    message << "Globe database R_PLANET=" << values.r_planet
            << " disagrees with model evaluator R_PLANET="
            << evaluator_r_planet;
    throw std::runtime_error(message.str());
  }
  if (!std::isfinite(evaluator_rhoav) ||
      std::abs(evaluator_rhoav - values.rhoav) >
          relative_tolerance * values.rhoav) {
    std::ostringstream message;
    message << "Globe database RHOAV=" << values.rhoav
            << " disagrees with model evaluator RHOAV=" << evaluator_rhoav;
    throw std::runtime_error(message.str());
  }
}

} // namespace specfem::globe::evaluator_impl

std::atomic_bool specfem::globe::ModelEvaluator::is_active_ = false;

specfem::globe::ModelEvaluator::Dimensions
specfem::globe::ModelEvaluator::dimensions() {
  const std::lock_guard<std::recursive_mutex> lock(
      specfem::globe::evaluator_impl::catalog_mutex);
  Dimensions result;
  globe_evaluator_dims(&result.ngllx, &result.nglly, &result.ngllz,
                       &result.n_sls);
  return result;
}

bool specfem::globe::ModelEvaluator::is_active() noexcept {
  return is_active_.load();
}

specfem::globe::ModelEvaluator::Scales
specfem::globe::ModelEvaluator::query_scales() {
  const std::lock_guard<std::recursive_mutex> lock(
      specfem::globe::evaluator_impl::catalog_mutex);
  Scales result;
  const int status =
      globe_evaluator_scales(&result.length, &result.density, &result.velocity);
  specfem::globe::evaluator_impl::require_ok(status, "query_scales");
  return result;
}

specfem::globe::ModelEvaluator::ModelEvaluator(
    const specfem::globe::ModelConfig &config,
    const specfem::globe::PlanetConstants &constants,
    const std::string &log_path)
    : constants_(constants) {
  const std::lock_guard<std::recursive_mutex> lock(
      specfem::globe::evaluator_impl::catalog_mutex);
  if (is_active_.load()) {
    throw std::runtime_error(
        "specfem::globe::ModelEvaluator: " +
        specfem::globe::evaluator_impl::describe_status(
            specfem::globe::evaluator_impl::status_already_initialized));
  }

  config.validate();
  specfem::globe::evaluator_impl::preflight_model_data(config);
  if (specfem::globe::planet_from_type(config.planet_type) !=
      constants_.planet()) {
    throw std::invalid_argument("specfem::globe::ModelEvaluator: MODEL_CONFIG "
                                "PLANET_TYPE disagrees with "
                                "the supplied PlanetConstants selection");
  }

#ifdef SPECFEM_ENABLE_MPI
  const int comm_f =
      static_cast<int>(MPI_Comm_c2f(specfem::MPI::communicator()));
#else
  constexpr int comm_f = 0;
#endif

  const int status = globe_evaluator_init(
      config.model_name.data(), static_cast<int>(config.model_name.size()),
      log_path.data(), static_cast<int>(log_path.size()), config.planet_type,
      config.nchunks, config.nex_xi, config.nex_eta, config.ellipticity ? 1 : 0,
      config.topography ? 1 : 0, config.oceans ? 1 : 0,
      config.attenuation ? 1 : 0, config.gravity ? 1 : 0,
      config.rotation ? 1 : 0, config.min_attenuation_period,
      config.max_attenuation_period, comm_f);

  if (status != specfem::globe::evaluator_impl::status_ok) {
    throw std::runtime_error(
        "specfem::globe::ModelEvaluator: failed to configure model '" +
        config.model_name +
        "': " + specfem::globe::evaluator_impl::describe_status(status));
  }

  owns_state_ = true;
  is_active_.store(true);
  try {
    specfem::globe::evaluator_impl::check_model_config(config);
    scales_ = query_scales();
    specfem::globe::evaluator_impl::check_scales(constants_, scales_.length,
                                                 scales_.density);
    constants_.check_radii(radii());
  } catch (...) {
    release();
    throw;
  }
}

specfem::globe::ModelEvaluator::~ModelEvaluator() { release(); }

specfem::globe::ModelEvaluator::ModelEvaluator(
    ModelEvaluator &&other) noexcept {
  const std::lock_guard<std::recursive_mutex> lock(
      specfem::globe::evaluator_impl::catalog_mutex);
  constants_ = std::move(other.constants_);
  scales_ = other.scales_;
  owns_state_ = std::exchange(other.owns_state_, false);
}

specfem::globe::ModelEvaluator &
specfem::globe::ModelEvaluator::operator=(ModelEvaluator &&other) noexcept {
  const std::lock_guard<std::recursive_mutex> lock(
      specfem::globe::evaluator_impl::catalog_mutex);
  if (this != &other) {
    release();
    constants_ = std::move(other.constants_);
    scales_ = other.scales_;
    owns_state_ = std::exchange(other.owns_state_, false);
  }
  return *this;
}

void specfem::globe::ModelEvaluator::release() noexcept {
  const std::lock_guard<std::recursive_mutex> lock(
      specfem::globe::evaluator_impl::catalog_mutex);
  if (owns_state_) {
    globe_evaluator_finalize();
    owns_state_ = false;
    is_active_.store(false);
  }
}

specfem::globe::PlanetConstants::Radii
specfem::globe::ModelEvaluator::radii() const {
  const std::lock_guard<std::recursive_mutex> lock(
      specfem::globe::evaluator_impl::catalog_mutex);
  specfem::globe::PlanetConstants::Radii result;
  const int status = globe_evaluator_radii(
      &result.r_icb, &result.r_cmb, &result.r_moho, &result.r_80, &result.r_220,
      &result.r_400, &result.r_670, &result.r_771, &result.r_ocean);
  specfem::globe::evaluator_impl::require_ok(status, "radii");
  result.validate(constants_.values().r_planet);
  return result;
}

double
specfem::globe::ModelEvaluator::reference_density(const double r_si) const {
  const std::lock_guard<std::recursive_mutex> lock(
      specfem::globe::evaluator_impl::catalog_mutex);
  const double r = specfem::globe::nondimensionalize(
                       specfem::units::Meters(r_si), constants_)
                       .raw();
  double rho = 0.0;
  const int status = globe_evaluator_reference_density(r, &rho);
  specfem::globe::evaluator_impl::require_ok(status, "reference_density");
  return specfem::globe::dimensionalize<specfem::units::KilogramPerCubicMeter>(
             specfem::units::Dimensionless(rho), constants_)
      .raw();
}

specfem::globe::ModelEvaluator::EllipticitySpline
specfem::globe::ModelEvaluator::ellipticity_spline() const {
  const std::lock_guard<std::recursive_mutex> lock(
      specfem::globe::evaluator_impl::catalog_mutex);
  int capacity = 0;
  globe_evaluator_reference_size(&capacity);
  if (capacity <= 0) {
    throw std::runtime_error(
        "specfem::globe::ModelEvaluator::ellipticity_spline: catalog returned "
        "an invalid reference-profile capacity");
  }

  EllipticitySpline result;
  result.radii.resize(static_cast<std::size_t>(capacity));
  result.values.resize(static_cast<std::size_t>(capacity));
  result.second_derivatives.resize(static_cast<std::size_t>(capacity));
  int size = 0;
  const int status = globe_evaluator_ellipticity_spline(
      capacity, &size, result.radii.data(), result.values.data(),
      result.second_derivatives.data());
  specfem::globe::evaluator_impl::require_ok(status, "ellipticity_spline");
  if (size <= 0 || size > capacity) {
    throw std::runtime_error(
        "specfem::globe::ModelEvaluator::ellipticity_spline: catalog returned "
        "an invalid spline size");
  }

  result.radii.resize(static_cast<std::size_t>(size));
  result.values.resize(static_cast<std::size_t>(size));
  result.second_derivatives.resize(static_cast<std::size_t>(size));
  const double length_scale = constants_.values().r_planet;
  for (double &radius : result.radii) {
    radius *= length_scale;
  }
  for (double &second_derivative : result.second_derivatives) {
    second_derivative /= length_scale * length_scale;
  }
  return result;
}

specfem::globe::ModelEvaluator::ElementProperties
specfem::globe::ModelEvaluator::evaluate_element(
    const int iregion_code, const int idoubling, const double rmin_si,
    const double rmax_si, const bool elem_in_crust, const bool elem_in_mantle,
    const std::vector<double> &xyz_si) const {
  const std::lock_guard<std::recursive_mutex> lock(
      specfem::globe::evaluator_impl::catalog_mutex);
  const std::size_t npoints = dimensions().points_per_element();
  if (xyz_si.size() != 3 * npoints) {
    std::ostringstream message;
    message << "specfem::globe::ModelEvaluator::evaluate_element: expected "
            << 3 * npoints << " coordinates but received " << xyz_si.size();
    throw std::invalid_argument(message.str());
  }

  const auto to_catalog_length = [this](const double value_si) {
    return specfem::globe::nondimensionalize(specfem::units::Meters(value_si),
                                             constants_)
        .raw();
  };
  const double rmin = to_catalog_length(rmin_si);
  const double rmax = to_catalog_length(rmax_si);
  std::vector<double> xyz(xyz_si.size());
  std::transform(xyz_si.begin(), xyz_si.end(), xyz.begin(), to_catalog_length);

  ElementProperties properties;
  properties.rho.resize(npoints);
  properties.vpv.resize(npoints);
  properties.vph.resize(npoints);
  properties.vsv.resize(npoints);
  properties.vsh.resize(npoints);
  properties.eta.resize(npoints);
  properties.vp_iso.resize(npoints);
  properties.vs_iso.resize(npoints);
  properties.qmu.resize(npoints);
  properties.qkappa.resize(npoints);
  properties.cij.resize(21 * npoints);
  properties.gc_prime.resize(npoints);
  properties.gs_prime.resize(npoints);

  int is_anisotropic = 0;
  const int status = globe_evaluator_get_element(
      iregion_code, idoubling, rmin, rmax, elem_in_crust ? 1 : 0,
      elem_in_mantle ? 1 : 0, xyz.data(), properties.rho.data(),
      properties.vpv.data(), properties.vph.data(), properties.vsv.data(),
      properties.vsh.data(), properties.eta.data(), properties.vp_iso.data(),
      properties.vs_iso.data(), properties.qmu.data(), properties.qkappa.data(),
      properties.cij.data(), properties.gc_prime.data(),
      properties.gs_prime.data(), &is_anisotropic);
  specfem::globe::evaluator_impl::require_ok(status, "evaluate_element");

  for (double &rho : properties.rho) {
    rho = specfem::globe::dimensionalize<specfem::units::KilogramPerCubicMeter>(
              specfem::units::Dimensionless(rho), constants_)
              .raw();
  }
  const auto scale_velocity = [this](double &value) {
    value *= scales_.velocity;
  };
  for (auto *values :
       { &properties.vpv, &properties.vph, &properties.vsv, &properties.vsh,
         &properties.vp_iso, &properties.vs_iso }) {
    std::for_each(values->begin(), values->end(), scale_velocity);
  }
  for (double &value : properties.cij) {
    value *= scales_.modulus();
  }
  properties.is_anisotropic = (is_anisotropic != 0);
  return properties;
}

specfem::globe::ModelEvaluator::ReferencePoint
specfem::globe::ModelEvaluator::prem_reference(const double r_si,
                                               const int idoubling,
                                               const int iregion_code) const {
  const std::lock_guard<std::recursive_mutex> lock(
      specfem::globe::evaluator_impl::catalog_mutex);
  ReferencePoint point;
  const double r = specfem::globe::nondimensionalize(
                       specfem::units::Meters(r_si), constants_)
                       .raw();
  const int status = globe_evaluator_prem_reference(
      r, idoubling, iregion_code, &point.rho, &point.vpv, &point.vph,
      &point.vsv, &point.vsh, &point.eta, &point.vp_iso, &point.vs_iso,
      &point.qkappa, &point.qmu);
  specfem::globe::evaluator_impl::require_ok(status, "prem_reference");

  point.rho =
      specfem::globe::dimensionalize<specfem::units::KilogramPerCubicMeter>(
          specfem::units::Dimensionless(point.rho), constants_)
          .raw();
  point.vpv *= scales_.velocity;
  point.vph *= scales_.velocity;
  point.vsv *= scales_.velocity;
  point.vsh *= scales_.velocity;
  point.vp_iso *= scales_.velocity;
  point.vs_iso *= scales_.velocity;
  return point;
}
