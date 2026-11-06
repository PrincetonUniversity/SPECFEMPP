#pragma once

#include "fk_injection/config.hpp"
#include "fk_injection/layered_model.hpp"
#include "specfem/plugin/injection_plugin.hpp"
#include <algorithm>
#include <array>
#include <stdexcept>

namespace fk_injection {

/**
 * @brief Elastic isotropic medium: the series of its rows and the quantities
 *        formed from them.
 *
 * Traction uses @f$\sigma_{rr} = \xi_1\sigma_{zz} - 4\xi_\mu d_x@f$ and
 * @f$\sigma_{\theta\theta} = b(\sigma_{rr} + \sigma_{zz})@f$, as SPECFEM3D;
 * with the factors 0 it is @f$\sigma_{rz}@f$ and @f$\sigma_{zz}@f$ on the
 * normal.
 */
struct ElasticMedium {
  /// @brief Depth-only series of an elastic row.
  enum Series : int {
    horizontal = 0, ///< Radial velocity
    vertical = 1,   ///< Vertical velocity
    sigma_zz = 2,   ///< @f$\omega p \sigma_{zz}@f$
    sigma_rz = 3,   ///< @f$-i \omega p \sigma_{rz}@f$
    dx = 4,         ///< @f$\omega p u_x@f$
  };
  static constexpr int number_of_series = 5;
  static constexpr specfem::plugin::Medium medium =
      specfem::plugin::Medium::elastic;
  static constexpr std::array<specfem::plugin::Quantity, 2> quantities = {
    specfem::plugin::Quantity::velocity, specfem::plugin::Quantity::traction
  };
  using ModelLayer = ElasticIsotropicLayer;

  /** @brief What sampling needs at one point. */
  struct Point {
    int row = 0;            ///< Row of the series table
    double offset = 0;      ///< Spline abscissa of step 0
    type_real xi1 = 0;      ///< First Lame-ratio factor
    type_real xim = 0;      ///< Shear Lame-ratio factor
    type_real bulk = 0;     ///< Bulk Lame-ratio factor
    type_real normal_x = 0; ///< x-component of the unit normal
    type_real normal_y = 0; ///< y-component of the unit normal
    type_real normal_z = 0; ///< z-component of the unit normal
  };

  /**
   * @brief A point's traction factors, from its layer, and its normal.
   * @param layer    Model layer of the point
   * @param normal_x x-component of the unit normal
   * @param normal_y y-component of the unit normal
   * @param normal_z z-component of the unit normal
   * @return The point, at row 0 and offset 0
   */
  static Point point(const ModelLayer &layer, type_real normal_x,
                     type_real normal_y, type_real normal_z) {
    const double density = static_cast<double>(layer.density);
    const double p_velocity = static_cast<double>(layer.p_velocity);
    const double s_velocity = static_cast<double>(layer.s_velocity);
    const double xi = (s_velocity * s_velocity) / (p_velocity * p_velocity);
    const double mu = density * s_velocity * s_velocity;
    const double lambda = density * p_velocity * p_velocity - 2.0 * mu;
    Point result;
    result.xi1 = static_cast<type_real>(1.0 - 2.0 * xi);
    result.xim = static_cast<type_real>((1.0 - xi) * mu);
    result.bulk = static_cast<type_real>(lambda / (2.0 * (lambda + mu)));
    result.normal_x = normal_x;
    result.normal_y = normal_y;
    result.normal_z = normal_z;
    return result;
  }

  /**
   * @brief Form @p quantity at a point from its row's series.
   *
   * Velocity is (cos_phi*horizontal, sin_phi*horizontal, vertical); traction
   * is the rotated stress on the point's normal.
   *
   * @param quantity Velocity or traction
   * @param point    Point
   * @param cos_phi  Cosine of the azimuth
   * @param sin_phi  Sine of the azimuth
   * @param evaluate Value of a @c Series at the point's abscissa
   * @return The 3 components
   */
  template <typename Evaluate>
  static std::array<double, 3> project(const specfem::plugin::Quantity quantity,
                                       const Point &point, const double cos_phi,
                                       const double sin_phi,
                                       const Evaluate &evaluate) {
    if (quantity == specfem::plugin::Quantity::velocity) {
      const double horizontal_value = evaluate(horizontal);
      return { cos_phi * horizontal_value, sin_phi * horizontal_value,
               evaluate(vertical) };
    }
    if (quantity != specfem::plugin::Quantity::traction)
      throw std::logic_error("ElasticMedium: unsupported quantity");
    const double sigma_zz_value = evaluate(sigma_zz);
    const double sigma_rz_value = evaluate(sigma_rz);
    const double dx_value = evaluate(dx);
    const double xi1 = static_cast<double>(point.xi1);
    const double xim = static_cast<double>(point.xim);
    const double bulk = static_cast<double>(point.bulk);
    const double nx = static_cast<double>(point.normal_x);
    const double ny = static_cast<double>(point.normal_y);
    const double nz = static_cast<double>(point.normal_z);

    const double sigma_rr = xi1 * sigma_zz_value - 4.0 * xim * dx_value;
    const double sigma_tt = bulk * (sigma_rr + sigma_zz_value);
    const double txx =
        sigma_rr * cos_phi * cos_phi + sigma_tt * sin_phi * sin_phi;
    const double txy = cos_phi * sin_phi * (sigma_rr - sigma_tt);
    const double txz = sigma_rz_value * cos_phi;
    const double tyy =
        sigma_rr * sin_phi * sin_phi + sigma_tt * cos_phi * cos_phi;
    const double tyz = sigma_rz_value * sin_phi;
    const double tzz = sigma_zz_value;
    return { txx * nx + txy * ny + txz * nz, txy * nx + tyy * ny + tyz * nz,
             txz * nx + tyz * ny + tzz * nz };
  }
};

/**
 * @brief Acoustic isotropic medium: the series of its rows and the quantities
 *        formed from them.
 *
 * Acoustic Stacey injection consumes the incident displacement and the
 * potential rate, as SPECFEM3D's acoustic FK coupling.
 */
struct AcousticMedium {
  /// @brief Depth-only series of an acoustic row.
  enum Series : int {
    horizontal = 0,     ///< Horizontal displacement
    vertical = 1,       ///< Vertical displacement
    potential_rate = 2, ///< @f$\dot\chi@f$
  };
  static constexpr int number_of_series = 3;
  static constexpr specfem::plugin::Medium medium =
      specfem::plugin::Medium::acoustic;
  static constexpr std::array<specfem::plugin::Quantity, 2> quantities = {
    specfem::plugin::Quantity::displacement,
    specfem::plugin::Quantity::potential_rate
  };
  using ModelLayer = AcousticLayer;

  /** @brief What sampling needs at one point. */
  struct Point {
    int row = 0;       ///< Row of the series table
    double offset = 0; ///< Spline abscissa of step 0
  };

  /**
   * @brief A point, which needs nothing of its layer or normal.
   * @return The point, at row 0 and offset 0
   */
  static Point point(const ModelLayer &, type_real, type_real, type_real) {
    return {};
  }

  /**
   * @brief Form @p quantity at a point from its row's series.
   *
   * Displacement is (cos_phi*horizontal, sin_phi*horizontal, vertical).
   *
   * @param quantity Displacement or potential rate
   * @param cos_phi  Cosine of the azimuth
   * @param sin_phi  Sine of the azimuth
   * @param evaluate Value of a @c Series at the point's abscissa
   * @return 3 components of displacement, or the potential rate first
   */
  template <typename Evaluate>
  static std::array<double, 3> project(const specfem::plugin::Quantity quantity,
                                       const Point &, const double cos_phi,
                                       const double sin_phi,
                                       const Evaluate &evaluate) {
    if (quantity == specfem::plugin::Quantity::potential_rate)
      return { evaluate(potential_rate), 0.0, 0.0 };
    if (quantity != specfem::plugin::Quantity::displacement)
      throw std::logic_error("AcousticMedium: unsupported quantity");
    const double horizontal_value = evaluate(horizontal);
    return { cos_phi * horizontal_value, sin_phi * horizontal_value,
             evaluate(vertical) };
  }
};

/**
 * @brief Whether @p Medium forms @p quantity.
 * @tparam Medium @c ElasticMedium or @c AcousticMedium
 * @param quantity Quantity
 * @return true if @p quantity is in @c Medium::quantities
 */
template <typename Medium>
constexpr bool serves(const specfem::plugin::Quantity quantity) {
  return std::find(Medium::quantities.begin(), Medium::quantities.end(),
                   quantity) != Medium::quantities.end();
}

/**
 * @brief Components of a quantity.
 * @param quantity Quantity
 * @return 1 for the potential rate, 3 otherwise
 */
constexpr int number_of_components(const specfem::plugin::Quantity quantity) {
  return quantity == specfem::plugin::Quantity::potential_rate ? 1 : 3;
}

/**
 * @brief Series a row of @p medium stores.
 * @param medium Acoustic or elastic
 * @return @c AcousticMedium::number_of_series or
 *         @c ElasticMedium::number_of_series
 */
constexpr int number_of_series(const specfem::plugin::Medium medium) {
  return medium == specfem::plugin::Medium::acoustic
             ? AcousticMedium::number_of_series
             : ElasticMedium::number_of_series;
}

} // namespace fk_injection
