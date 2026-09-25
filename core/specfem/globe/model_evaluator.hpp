#pragma once

#include "specfem/globe/model_config.hpp"
#include "specfem/globe/planet_constants.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace specfem::globe {

/**
 * @brief Move-only RAII boundary around the SPECFEM3D_GLOBE model catalog.
 *
 * Construction replays the database's model configuration into the Fortran
 * catalog. All public coordinates and material values are SI; conversion to
 * and from the catalog's non-dimensional convention is confined to this
 * wrapper.
 *
 * Only one instance may own the process-global Fortran catalog at a time, and
 * evaluation is setup-only and single-threaded because catalog routines retain
 * module and `save` state.
 */
class ModelEvaluator {
public:
  /** @brief Quadrature dimensions compiled into the model catalog. */
  struct Dimensions {
    int ngllx = 0;
    int nglly = 0;
    int ngllz = 0;
    int n_sls = 0;

    /** @brief Number of GLL points in one element. */
    [[nodiscard]] std::size_t points_per_element() const {
      return static_cast<std::size_t>(ngllx) * static_cast<std::size_t>(nglly) *
             static_cast<std::size_t>(ngllz);
    }
  };

  /** @brief SI material properties at every GLL point of one element. */
  struct ElementProperties {
    std::vector<double> rho; ///< Density, kg/m^3.
    std::vector<double> vpv; ///< Vertical P velocity, m/s.
    std::vector<double> vph; ///< Horizontal P velocity, m/s.
    std::vector<double> vsv; ///< Vertical S velocity, m/s.
    std::vector<double> vsh; ///< Horizontal S velocity, m/s.
    std::vector<double> eta; ///< Dimensionless anisotropy parameter.

    std::vector<double> vp_iso; ///< Isotropic P velocity, m/s.
    std::vector<double> vs_iso; ///< Isotropic S velocity, m/s.
    std::vector<double> qmu;    ///< Dimensionless shear quality factor.
    std::vector<double> qkappa; ///< Dimensionless bulk quality factor.

    /** @brief Elastic coefficients in Pa, point-major with 21 per point. */
    std::vector<double> cij;

    std::vector<double> gc_prime; ///< Dimensionless azimuthal anisotropy.
    std::vector<double> gs_prime; ///< Dimensionless azimuthal anisotropy.
    bool is_anisotropic = false;
  };

  /** @brief SI values from the catalog's direct PREM reference path. */
  struct ReferencePoint {
    double rho = 0.0;
    double vpv = 0.0;
    double vph = 0.0;
    double vsv = 0.0;
    double vsh = 0.0;
    double eta = 0.0;
    double vp_iso = 0.0;
    double vs_iso = 0.0;
    double qkappa = 0.0;
    double qmu = 0.0;
  };

  /**
   * @brief Configure the catalog from the resolved model selection.
   * @param config Opaque `MODEL_CONFIG` values read from the database.
   * @param log_path Optional catalog log path; empty redirects to `/dev/null`.
   */
  explicit ModelEvaluator(const ModelConfig &config,
                          const std::string &log_path = "");

  /**
   * @brief Validate transient database constants against the model catalog.
   * @param config Opaque `MODEL_CONFIG` values read from the database.
   * @param planet_constants Selected planet's SI constants, discarded after
   * validation.
   * @param log_path Optional catalog log path; empty redirects to `/dev/null`.
   */
  static void
  validate_database_constants(const ModelConfig &config,
                              const PlanetConstants &planet_constants,
                              const std::string &log_path = "");

  ~ModelEvaluator();

  ModelEvaluator(const ModelEvaluator &) = delete;
  ModelEvaluator &operator=(const ModelEvaluator &) = delete;
  ModelEvaluator(ModelEvaluator &&other) noexcept;
  ModelEvaluator &operator=(ModelEvaluator &&other) noexcept;

  /**
   * @brief Evaluate one element and return only SI material values.
   * @param iregion_code Globe radial region code.
   * @param idoubling Globe radial-zone flag.
   * @param rmin_si Minimum shell radius, m.
   * @param rmax_si Maximum shell radius, m.
   * @param elem_in_crust Whether the element intersects the crust model.
   * @param elem_in_mantle Whether the element intersects the mantle model.
   * @param xyz_si Point-major Cartesian coordinates, m.
   */
  [[nodiscard]] ElementProperties
  evaluate_element(int iregion_code, int idoubling, double rmin_si,
                   double rmax_si, bool elem_in_crust, bool elem_in_mantle,
                   const std::vector<double> &xyz_si) const;

  /** @brief Quadrature dimensions compiled into the catalog. */
  [[nodiscard]] static Dimensions dimensions();

  /** @brief Whether any wrapper currently owns the Fortran catalog. */
  [[nodiscard]] static bool is_active() noexcept;

  /** @brief Evaluate the direct PREM reference path in SI. Test use only. */
  [[nodiscard]] ReferencePoint prem_reference(double r_si, int idoubling,
                                              int iregion_code) const;

private:
  struct Scales {
    double length = 0.0;
    double density = 0.0;
    double velocity = 0.0;

    [[nodiscard]] double to_catalog_length(double value_si) const {
      return value_si / length;
    }

    [[nodiscard]] double to_si_density(double value) const {
      return value * density;
    }

    [[nodiscard]] double to_si_velocity(double value) const {
      return value * velocity;
    }

    [[nodiscard]] double to_si_modulus(double value) const {
      return value * density * velocity * velocity;
    }
  };

  [[nodiscard]] static Scales query_scales();
  [[nodiscard]] static std::vector<double>
  query_planet_values(int schema_version, std::size_t number_of_values);
  void release() noexcept;

  Scales scales_;
  bool owns_state_ = false;

  static bool is_active_;
};

} // namespace specfem::globe
