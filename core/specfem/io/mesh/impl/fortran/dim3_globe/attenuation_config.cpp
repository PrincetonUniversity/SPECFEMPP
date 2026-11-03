#include "specfem/io/mesh/impl/fortran/dim3_globe/attenuation_config.hpp"
#include "specfem/attenuation/compute_tau_sigma.hpp"
#include "specfem/constants.hpp"
#include "specfem/io/mesh/impl/fortran/dim3_globe/globe_codes.hpp"
#include "specfem/logger.hpp"
#include "specfem/units.hpp"
#include "specfem/utilities/band.hpp"

#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

specfem::mesh::attenuation_config
specfem::io::mesh::impl::fortran::dim3_globe::make_attenuation_config(
    const specfem::globe::ModelConfig &model_config,
    const std::vector<bool> &model_flags,
    const specfem::attenuation::Setup &attenuation_setup) {
  namespace reader = specfem::io::mesh::impl::fortran::dim3_globe;
  using specfem::units::unit_symbols::Hz;

  specfem::mesh::attenuation_config config;

  if (attenuation_setup.enabled && !model_config.attenuation) {
    throw std::runtime_error(
        "Attenuation is enabled in the configuration, but the globe mesh "
        "database was meshed with ATTENUATION = .false., so the model oracle "
        "provides no Q. Re-mesh with ATTENUATION = .true. or disable "
        "attenuation.");
  }
  if (!attenuation_setup.enabled) {
    if (model_config.attenuation) {
      specfem::Logger::warning(
          "The globe mesh database was meshed with ATTENUATION = .true., but "
          "attenuation is disabled in the configuration; running elastic.");
    }
    return config;
  }

  if (model_flags.size() != reader::model_flag_count) {
    throw std::runtime_error("Globe database must carry " +
                             std::to_string(reader::model_flag_count) +
                             " model flags, found " +
                             std::to_string(model_flags.size()));
  }
  // Q from these models has not yet been checked to be band-independent and
  // to match the mesher's get_model output.
  if (model_flags[reader::model_flag_attenuation_3d_berkeley] ||
      model_flags[reader::model_flag_attenuation_gll]) {
    throw std::runtime_error(
        "Globe attenuation with ATTENUATION_3D_BERKELEY or ATTENUATION_GLL Q "
        "models is not yet supported.");
  }

  // The oracle and the mesher use the database band, so the solver band must
  // match it for Q and the relaxation mechanisms to describe the same model.
  const specfem::utilities::Band<specfem::units::Hertz> database_band(
      (1.0 / model_config.max_attenuation_period) * Hz,
      (1.0 / model_config.min_attenuation_period) * Hz);
  const auto &band = attenuation_setup.band;
  constexpr double relative_tolerance = 1.0e-6;
  const auto differs = [&](const double value, const double expected) {
    return std::abs(value - expected) > relative_tolerance * std::abs(expected);
  };
  if (differs(band.min.raw(), database_band.min.raw()) ||
      differs(band.max.raw(), database_band.max.raw())) {
    std::ostringstream message;
    message << std::setprecision(10) << "attenuation-frequency-band ["
            << band.min.raw() << ", " << band.max.raw()
            << "] Hz does not match the globe mesh database band ["
            << database_band.min.raw() << ", " << database_band.max.raw()
            << "] Hz (attenuation periods ["
            << model_config.min_attenuation_period << ", "
            << model_config.max_attenuation_period << "] s).";
    throw std::runtime_error(message.str());
  }

  if (!attenuation_setup.f0.has_value()) {
    throw std::runtime_error(
        "Globe attenuation requires an explicit reference frequency");
  }

  config.enabled = true;
  config.f0 = attenuation_setup.f0.value();
  config.band = band;
  config.tau_sigma =
      specfem::attenuation::compute_tau_sigma<specfem::constants::N_SLS>(band);
  // SPECFEM3D_GLOBE implements Q_mu attenuation only.
  config.bulk_attenuation = false;
  return config;
}
