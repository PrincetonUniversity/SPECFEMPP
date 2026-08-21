#include "specfem/globe_model/evaluator.hpp"

#include "specfem/mpi.hpp"

#include <sstream>
#include <stdexcept>
#include <string>

// The raw Fortran ABI is deliberately confined to this translation unit so it
// does not leak into the public header. Declared by hand rather than by
// including the Fortran side's globe_model_oracle.h, to keep core/ free of an
// include path into fortran/.
extern "C" {

void globe_oracle_dims(int *ngllx, int *nglly, int *ngllz, int *n_sls);

int globe_oracle_init(const char *model_name, int name_len,
                      const char *imain_path, int imain_path_len,
                      int planet_type, int nchunks, int nex_xi, int nex_eta,
                      int ellipticity, int topography, int oceans,
                      int attenuation, int gravity, int rotation,
                      double min_attenuation_period,
                      double max_attenuation_period, int comm_f);

int globe_oracle_scales(double *length_scale, double *density_scale,
                        double *velocity_scale);

int globe_oracle_finalize(void);

int globe_oracle_get_element(int iregion_code, int idoubling, double rmin_si,
                             double rmax_si, int elem_in_crust,
                             int elem_in_mantle, const double *xyz_si,
                             double *rho, double *vpv, double *vph, double *vsv,
                             double *vsh, double *eta, double *vp_iso,
                             double *vs_iso, double *qmu, double *qkappa,
                             double *cij, double *gc_prime, double *gs_prime,
                             int *is_anisotropic);

int globe_oracle_prem_reference(double r_si, int idoubling, int iregion_code,
                                double *rho, double *vpv, double *vph,
                                double *vsv, double *vsh, double *eta,
                                double *vp_iso, double *vs_iso, double *qkappa,
                                double *qmu);
}

namespace specfem {
namespace globe_model_impl {

// Mirrors globe_oracle_par in fortran/globe_oracle/globe_model_oracle.F90.
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
           "specfem::globe_model::Evaluator may exist at a time because the "
           "catalog's configuration lives in Fortran module state";
  case status_not_initialized:
    return "the globe model catalog has not been configured";
  case status_unsupported_model:
    return "this model requires per-GLL indexing into the mesher's own "
           "discretization (MODEL_GLL, CEM, HETEROGEN_3D_MANTLE, "
           "ATTENUATION_GLL or EMC) and cannot be evaluated from a position "
           "alone";
  case status_imain_open_failed:
    return "could not open the log file for the globe model catalog";
  case status_bad_argument:
    return "the globe model catalog rejected an argument as out of range";
  default: {
    std::ostringstream message;
    message << "unrecognized status code " << status;
    return message.str();
  }
  }
}

} // namespace globe_model_impl
} // namespace specfem

bool specfem::globe_model::Evaluator::is_active_ = false;

specfem::globe_model::Dims specfem::globe_model::Evaluator::dims() {
  specfem::globe_model::Dims result;
  globe_oracle_dims(&result.ngllx, &result.nglly, &result.ngllz, &result.n_sls);
  return result;
}

bool specfem::globe_model::Evaluator::is_active() { return is_active_; }

specfem::globe_model::Scales specfem::globe_model::Evaluator::scales() {
  specfem::globe_model::Scales result;

  const int status =
      globe_oracle_scales(&result.length, &result.density, &result.velocity);

  if (status != specfem::globe_model_impl::status_ok) {
    throw std::runtime_error(
        "specfem::globe_model::Evaluator::scales: " +
        specfem::globe_model_impl::describe_status(status));
  }

  return result;
}

specfem::globe_model::Evaluator::Evaluator(
    const specfem::globe_model::ModelConfig &config) {

  if (is_active_) {
    throw std::runtime_error(
        "specfem::globe_model::Evaluator: " +
        specfem::globe_model_impl::describe_status(
            specfem::globe_model_impl::status_already_initialized));
  }

  // The catalog broadcasts through a Fortran communicator handle, so the C
  // handle has to be translated. specfem::MPI::communicator() is a real
  // MPI_Comm here: the globe path requires SPECFEM_ENABLE_MPI.
  const int comm_f =
      static_cast<int>(MPI_Comm_c2f(specfem::MPI::communicator()));

  const int status = globe_oracle_init(
      config.model_name.data(), static_cast<int>(config.model_name.size()),
      config.log_path.data(), static_cast<int>(config.log_path.size()),
      config.planet_type, config.nchunks, config.nex_xi, config.nex_eta,
      config.ellipticity ? 1 : 0, config.topography ? 1 : 0,
      config.oceans ? 1 : 0, config.attenuation ? 1 : 0, config.gravity ? 1 : 0,
      config.rotation ? 1 : 0, config.min_attenuation_period,
      config.max_attenuation_period, comm_f);

  if (status != specfem::globe_model_impl::status_ok) {
    std::ostringstream message;
    message << "specfem::globe_model::Evaluator: failed to configure model '"
            << config.model_name
            << "': " << specfem::globe_model_impl::describe_status(status);
    throw std::runtime_error(message.str());
  }

  is_active_ = true;
}

specfem::globe_model::Evaluator::~Evaluator() {
  // Must clear the Fortran-side guard too, not just this mirror of it: the
  // catalog's `is_initialized` is what globe_oracle_init checks, so skipping
  // this would leave the process permanently unable to configure another model.
  globe_oracle_finalize();
  is_active_ = false;
}

specfem::globe_model::ElementProperties
specfem::globe_model::Evaluator::evaluate_element(
    const int iregion_code, const int idoubling, const double rmin_si,
    const double rmax_si, const bool elem_in_crust, const bool elem_in_mantle,
    const std::vector<double> &xyz_si) const {

  const auto quadrature = dims();
  const std::size_t npoints = quadrature.points_per_element();

  if (xyz_si.size() != 3 * npoints) {
    std::ostringstream message;
    message << "specfem::globe_model::Evaluator::evaluate_element: expected "
            << 3 * npoints << " coordinates (3 x " << npoints
            << " GLL points) but received " << xyz_si.size();
    throw std::invalid_argument(message.str());
  }

  specfem::globe_model::ElementProperties properties;
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

  const int status = globe_oracle_get_element(
      iregion_code, idoubling, rmin_si, rmax_si, elem_in_crust ? 1 : 0,
      elem_in_mantle ? 1 : 0, xyz_si.data(), properties.rho.data(),
      properties.vpv.data(), properties.vph.data(), properties.vsv.data(),
      properties.vsh.data(), properties.eta.data(), properties.vp_iso.data(),
      properties.vs_iso.data(), properties.qmu.data(), properties.qkappa.data(),
      properties.cij.data(), properties.gc_prime.data(),
      properties.gs_prime.data(), &is_anisotropic);

  if (status != specfem::globe_model_impl::status_ok) {
    throw std::runtime_error(
        "specfem::globe_model::Evaluator::evaluate_element: " +
        specfem::globe_model_impl::describe_status(status));
  }

  properties.is_anisotropic = (is_anisotropic != 0);

  return properties;
}

specfem::globe_model::ReferencePoint
specfem::globe_model::prem_reference(const double r_si, const int idoubling,
                                     const int iregion_code) {

  specfem::globe_model::ReferencePoint point{};

  const int status = globe_oracle_prem_reference(
      r_si, idoubling, iregion_code, &point.rho, &point.vpv, &point.vph,
      &point.vsv, &point.vsh, &point.eta, &point.vp_iso, &point.vs_iso,
      &point.qkappa, &point.qmu);

  if (status != specfem::globe_model_impl::status_ok) {
    throw std::runtime_error(
        "specfem::globe_model::prem_reference: " +
        specfem::globe_model_impl::describe_status(status));
  }

  return point;
}
