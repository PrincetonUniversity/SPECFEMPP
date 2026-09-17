#include "specfem/io/globe_model.hpp"

#include "specfem/mpi.hpp"
#include "specfem/units.hpp"
#include "specfem/utilities/dimensionalization.hpp"

#include <algorithm>
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
int globe_evaluator_radii(double *r_icb, double *r_cmb, double *r_moho,
                          double *r_80, double *r_220, double *r_400,
                          double *r_670, double *r_771, double *r_ocean);
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

namespace specfem::io::globe_model_impl {

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
           "specfem::io::globe_model may own its Fortran module state";
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
    throw std::runtime_error("specfem::io::globe_model::" + operation + ": " +
                             describe_status(status));
  }
}

} // namespace specfem::io::globe_model_impl

bool specfem::io::globe_model::is_active_ = false;

specfem::io::globe_model::Dimensions specfem::io::globe_model::dimensions() {
  Dimensions result;
  globe_evaluator_dims(&result.ngllx, &result.nglly, &result.ngllz,
                       &result.n_sls);
  return result;
}

bool specfem::io::globe_model::is_active() noexcept { return is_active_; }

specfem::io::globe_model::Scales specfem::io::globe_model::query_scales() {
  Scales result;
  const int status =
      globe_evaluator_scales(&result.length, &result.density, &result.velocity);
  specfem::io::globe_model_impl::require_ok(status, "query_scales");
  return result;
}

specfem::io::globe_model::globe_model(
    const specfem::io::GlobeModelConfig &config,
    const specfem::constants::PlanetConstants &constants,
    const std::string &log_path)
    : constants_(constants) {
  if (is_active_) {
    throw std::runtime_error(
        "specfem::io::globe_model: " +
        specfem::io::globe_model_impl::describe_status(
            specfem::io::globe_model_impl::status_already_initialized));
  }

  config.validate();
  if (specfem::constants::planet_from_type(config.planet_type) !=
      constants_.planet()) {
    throw std::invalid_argument(
        "specfem::io::globe_model: MODEL_CONFIG PLANET_TYPE disagrees with "
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

  if (status != specfem::io::globe_model_impl::status_ok) {
    throw std::runtime_error(
        "specfem::io::globe_model: failed to configure model '" +
        config.model_name +
        "': " + specfem::io::globe_model_impl::describe_status(status));
  }

  owns_state_ = true;
  is_active_ = true;
  try {
    scales_ = query_scales();
    specfem::constants::check_database_values(constants_, scales_.length,
                                              scales_.density);
  } catch (...) {
    release();
    throw;
  }
}

specfem::io::globe_model::~globe_model() { release(); }

specfem::io::globe_model::globe_model(globe_model &&other) noexcept
    : constants_(std::move(other.constants_)), scales_(other.scales_),
      owns_state_(std::exchange(other.owns_state_, false)) {}

specfem::io::globe_model &
specfem::io::globe_model::operator=(globe_model &&other) noexcept {
  if (this != &other) {
    release();
    constants_ = std::move(other.constants_);
    scales_ = other.scales_;
    owns_state_ = std::exchange(other.owns_state_, false);
  }
  return *this;
}

void specfem::io::globe_model::release() noexcept {
  if (owns_state_) {
    globe_evaluator_finalize();
    owns_state_ = false;
    is_active_ = false;
  }
}

specfem::constants::PlanetConstants::Radii
specfem::io::globe_model::radii() const {
  specfem::constants::PlanetConstants::Radii result;
  const int status = globe_evaluator_radii(
      &result.r_icb, &result.r_cmb, &result.r_moho, &result.r_80, &result.r_220,
      &result.r_400, &result.r_670, &result.r_771, &result.r_ocean);
  specfem::io::globe_model_impl::require_ok(status, "radii");
  result.validate(constants_.values().r_planet);
  return result;
}

specfem::io::globe_model::ElementProperties
specfem::io::globe_model::evaluate_element(
    const int iregion_code, const int idoubling, const double rmin_si,
    const double rmax_si, const bool elem_in_crust, const bool elem_in_mantle,
    const std::vector<double> &xyz_si) const {
  const std::size_t npoints = dimensions().points_per_element();
  if (xyz_si.size() != 3 * npoints) {
    std::ostringstream message;
    message << "specfem::io::globe_model::evaluate_element: expected "
            << 3 * npoints << " coordinates but received " << xyz_si.size();
    throw std::invalid_argument(message.str());
  }

  const auto to_catalog_length = [this](const double value_si) {
    return specfem::utilities::nondimensionalize(
               specfem::units::Meters(value_si), constants_)
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
  specfem::io::globe_model_impl::require_ok(status, "evaluate_element");

  for (double &rho : properties.rho) {
    rho = specfem::utilities::dimensionalize<
              specfem::units::KilogramPerCubicMeter>(
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

specfem::io::globe_model::ReferencePoint
specfem::io::globe_model::prem_reference(const double r_si, const int idoubling,
                                         const int iregion_code) const {
  ReferencePoint point;
  const double r = specfem::utilities::nondimensionalize(
                       specfem::units::Meters(r_si), constants_)
                       .raw();
  const int status = globe_evaluator_prem_reference(
      r, idoubling, iregion_code, &point.rho, &point.vpv, &point.vph,
      &point.vsv, &point.vsh, &point.eta, &point.vp_iso, &point.vs_iso,
      &point.qkappa, &point.qmu);
  specfem::io::globe_model_impl::require_ok(status, "prem_reference");

  point.rho =
      specfem::utilities::dimensionalize<specfem::units::KilogramPerCubicMeter>(
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
