#include "specfem/globe/model_evaluator.hpp"

#include "specfem/mpi.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

extern "C" {
void globe_evaluator_dims(int *ngllx, int *nglly, int *ngllz, int *n_sls);
int globe_evaluator_init(const char *model_name, int name_len,
                         const char *imain_path, int imain_path_len,
                         int planet_type, int nchunks, int nex_xi, int nex_eta,
                         int ellipticity, int topography, int oceans,
                         int attenuation, int gravity, int rotation,
                         double min_attenuation_period,
                         double max_attenuation_period, int comm_f);
int globe_evaluator_scales(double *length_scale, double *density_scale,
                           double *velocity_scale);
int globe_evaluator_planet_values(int schema_version, int number_of_values,
                                  double *values);
int globe_evaluator_finalize(void);
int globe_evaluator_get_element(int iregion_code, int idoubling, double rmin,
                                double rmax, int elem_in_crust,
                                int elem_in_mantle, const double *xyz,
                                double *rho, double *vpv, double *vph,
                                double *vsv, double *vsh, double *eta,
                                double *vp_iso, double *vs_iso, double *qmu,
                                double *qkappa, double *cij, double *gc_prime,
                                double *gs_prime, int *is_anisotropic);
int globe_evaluator_prem_reference(double r, int idoubling, int iregion_code,
                                   double *rho, double *vpv, double *vph,
                                   double *vsv, double *vsh, double *eta,
                                   double *vp_iso, double *vs_iso,
                                   double *qkappa, double *qmu);
}

namespace specfem::globe::evaluator_impl {

constexpr int status_ok = 0;
constexpr int status_already_initialized = 1;
constexpr int status_not_initialized = 2;
constexpr int status_unsupported_model = 3;
constexpr int status_imain_open_failed = 4;
constexpr int status_bad_argument = 5;

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

} // namespace specfem::globe::evaluator_impl

bool specfem::globe::ModelEvaluator::is_active_ = false;

specfem::globe::ModelEvaluator::Dimensions
specfem::globe::ModelEvaluator::dimensions() {
  Dimensions result;
  globe_evaluator_dims(&result.ngllx, &result.nglly, &result.ngllz,
                       &result.n_sls);
  return result;
}

bool specfem::globe::ModelEvaluator::is_active() noexcept { return is_active_; }

specfem::globe::ModelEvaluator::Scales
specfem::globe::ModelEvaluator::query_scales() {
  Scales result;
  const int status =
      globe_evaluator_scales(&result.length, &result.density, &result.velocity);
  specfem::globe::evaluator_impl::require_ok(status, "query_scales");
  return result;
}

specfem::globe::ModelEvaluator::ModelEvaluator(
    const specfem::globe::ModelConfig &config, const std::string &log_path) {
  if (is_active_) {
    throw std::runtime_error(
        "specfem::globe::ModelEvaluator: " +
        specfem::globe::evaluator_impl::describe_status(
            specfem::globe::evaluator_impl::status_already_initialized));
  }

  config.validate();

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
  is_active_ = true;
  try {
    scales_ = query_scales();
    if (!std::isfinite(scales_.length) || scales_.length <= 0.0 ||
        !std::isfinite(scales_.density) || scales_.density <= 0.0 ||
        !std::isfinite(scales_.velocity) || scales_.velocity <= 0.0) {
      throw std::runtime_error(
          "Globe model evaluator returned invalid dimensional scales");
    }
  } catch (...) {
    release();
    throw;
  }
}

void specfem::globe::ModelEvaluator::validate_database_constants(
    const specfem::globe::ModelConfig &config,
    const specfem::globe::PlanetConstants &planet_constants,
    const std::string &log_path) {
  const ModelEvaluator evaluator(config, log_path);

  if (specfem::globe::planet_from_type(config.planet_type) !=
      planet_constants.planet()) {
    throw std::invalid_argument(
        "specfem::globe::ModelEvaluator: MODEL_CONFIG PLANET_TYPE disagrees "
        "with the supplied PlanetConstants selection");
  }

  planet_constants.check_catalog_values(query_planet_values(
      planet_constants.schema_version(), planet_constants.values().size()));
}

specfem::globe::ModelEvaluator::~ModelEvaluator() { release(); }

specfem::globe::ModelEvaluator::ModelEvaluator(ModelEvaluator &&other) noexcept
    : scales_(other.scales_),
      owns_state_(std::exchange(other.owns_state_, false)) {}

specfem::globe::ModelEvaluator &
specfem::globe::ModelEvaluator::operator=(ModelEvaluator &&other) noexcept {
  if (this != &other) {
    release();
    scales_ = other.scales_;
    owns_state_ = std::exchange(other.owns_state_, false);
  }
  return *this;
}

void specfem::globe::ModelEvaluator::release() noexcept {
  if (owns_state_) {
    globe_evaluator_finalize();
    owns_state_ = false;
    is_active_ = false;
  }
}

std::vector<double> specfem::globe::ModelEvaluator::query_planet_values(
    const int schema_version, const std::size_t number_of_values) {
  if (number_of_values >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error(
        "Planet schema value count exceeds Fortran integer range");
  }

  std::vector<double> values(number_of_values);
  const int status = globe_evaluator_planet_values(
      schema_version, static_cast<int>(number_of_values), values.data());
  specfem::globe::evaluator_impl::require_ok(status, "query_planet_values");
  return values;
}

specfem::globe::ModelEvaluator::ElementProperties
specfem::globe::ModelEvaluator::evaluate_element(
    const int iregion_code, const int idoubling, const double rmin_si,
    const double rmax_si, const bool elem_in_crust, const bool elem_in_mantle,
    const std::vector<double> &xyz_si) const {
  const std::size_t npoints = dimensions().points_per_element();
  if (xyz_si.size() != 3 * npoints) {
    std::ostringstream message;
    message << "specfem::globe::ModelEvaluator::evaluate_element: expected "
            << 3 * npoints << " coordinates but received " << xyz_si.size();
    throw std::invalid_argument(message.str());
  }

  const double rmin = scales_.to_catalog_length(rmin_si);
  const double rmax = scales_.to_catalog_length(rmax_si);
  std::vector<double> xyz(xyz_si.size());
  std::transform(xyz_si.begin(), xyz_si.end(), xyz.begin(),
                 [this](const double value_si) {
                   return scales_.to_catalog_length(value_si);
                 });

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
    rho = scales_.to_si_density(rho);
  }
  const auto scale_velocity = [this](double &value) {
    value = scales_.to_si_velocity(value);
  };
  for (auto *values :
       { &properties.vpv, &properties.vph, &properties.vsv, &properties.vsh,
         &properties.vp_iso, &properties.vs_iso }) {
    std::for_each(values->begin(), values->end(), scale_velocity);
  }
  for (double &value : properties.cij) {
    value = scales_.to_si_modulus(value);
  }
  properties.is_anisotropic = (is_anisotropic != 0);
  return properties;
}

specfem::globe::ModelEvaluator::ReferencePoint
specfem::globe::ModelEvaluator::prem_reference(const double r_si,
                                               const int idoubling,
                                               const int iregion_code) const {
  ReferencePoint point;
  const double r = scales_.to_catalog_length(r_si);
  const int status = globe_evaluator_prem_reference(
      r, idoubling, iregion_code, &point.rho, &point.vpv, &point.vph,
      &point.vsv, &point.vsh, &point.eta, &point.vp_iso, &point.vs_iso,
      &point.qkappa, &point.qmu);
  specfem::globe::evaluator_impl::require_ok(status, "prem_reference");

  point.rho = scales_.to_si_density(point.rho);
  point.vpv = scales_.to_si_velocity(point.vpv);
  point.vph = scales_.to_si_velocity(point.vph);
  point.vsv = scales_.to_si_velocity(point.vsv);
  point.vsh = scales_.to_si_velocity(point.vsh);
  point.vp_iso = scales_.to_si_velocity(point.vp_iso);
  point.vs_iso = scales_.to_si_velocity(point.vs_iso);
  return point;
}
