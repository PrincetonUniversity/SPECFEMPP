#pragma once

#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>

/**
 * @brief Analytic plane wave in a homogeneous, isotropic elastic whole space.
 *
 * Depends only on the standard library, so the physics can be read and tested
 * without the plugin SDK.
 */
namespace plane_wave {

using Vector = std::array<double, 3>;
using Tensor = std::array<Vector, 3>;

/** @brief Homogeneous isotropic elastic material. */
struct Material {
  double rho = 0; ///< Density
  double vp = 0;  ///< P-wave speed
  double vs = 0;  ///< S-wave speed

  /** @brief Shear modulus. @return \f$ \rho v_s^2 \f$ */
  double mu() const { return rho * vs * vs; }

  /** @brief First Lamé parameter. @return \f$ \rho (v_p^2 - 2 v_s^2) \f$ */
  double lambda() const { return rho * (vp * vp - 2 * vs * vs); }
};

/** @brief Wave type: compressional or vertically polarized shear. */
enum class WaveType { p, sv };

/**
 * @brief Displacement waveform \f$ f(\tau) \f$ and its derivatives.
 *
 * - Gaussian: \f$ f = e^{-(\tau/h)^2} \f$, with half-duration \f$ h \f$.
 * - Ricker: \f$ f = (1 - 2\pi^2 f_0^2 \tau^2)\, e^{-\pi^2 f_0^2 \tau^2} \f$,
 *   with dominant frequency \f$ f_0 \f$.
 */
class SourceTimeFunction {
public:
  /**
   * @brief Gaussian waveform.
   * @param half_duration Half-duration \f$ h \f$
   * @return Waveform
   */
  static SourceTimeFunction gaussian(const double half_duration) {
    return { Kind::gaussian, 1 / (half_duration * half_duration) };
  }

  /**
   * @brief Ricker waveform.
   * @param f0 Dominant frequency
   * @return Waveform
   */
  static SourceTimeFunction ricker(const double f0) {
    return { Kind::ricker, std::numbers::pi * std::numbers::pi * f0 * f0 };
  }

  /**
   * @brief Derivative of the waveform, computed analytically.
   * @param tau Delayed time \f$ \tau \f$
   * @param order 0, 1 or 2
   * @return \f$ f^{(order)}(\tau) \f$
   */
  double derivative(const double tau, const int order) const {
    // a = 1/h^2 (gaussian) or pi^2 f0^2 (ricker); both decay as exp(-a tau^2)
    const double a = scale_;
    const double t2 = tau * tau;
    const double e = std::exp(-a * t2);
    if (kind_ == Kind::gaussian) {
      switch (order) {
      case 0:
        return e;
      case 1:
        return -2 * a * tau * e;
      case 2:
        return (4 * a * a * t2 - 2 * a) * e;
      }
    } else {
      switch (order) {
      case 0:
        return (1 - 2 * a * t2) * e;
      case 1:
        return 2 * a * tau * (2 * a * t2 - 3) * e;
      case 2:
        return 2 * a * (-4 * a * a * t2 * t2 + 12 * a * t2 - 3) * e;
      }
    }
    throw std::invalid_argument("SourceTimeFunction: order must be 0, 1 or 2");
  }

private:
  enum class Kind { gaussian, ricker };

  SourceTimeFunction(const Kind kind, const double scale)
      : kind_(kind), scale_(scale) {}

  Kind kind_;
  double scale_;
};

/**
 * @brief Plane wave \f$ u = A\,\hat{d}\,f(\tau) \f$, with
 * \f$ \tau = t - t_0 - \hat{k} \cdot (x - x_0)/c \f$.
 *
 * Conventions (z up):
 * - direction \f$ \hat{k} = (\sin\theta\cos\phi, \sin\theta\sin\phi,
 *   \cos\theta) \f$, with take-off angle \f$ \theta \f$ from +z and azimuth
 *   \f$ \phi \f$ from +x toward +y;
 * - polarization: P has \f$ \hat{d} = \hat{k} \f$ and \f$ c = v_p \f$; SV has
 *   \f$ \hat{d} = \partial\hat{k}/\partial\theta = (\cos\theta\cos\phi,
 *   \cos\theta\sin\phi, -\sin\theta) \f$ and \f$ c = v_s \f$.
 */
class PlaneWave {
public:
  /**
   * @brief Plane wave of a given type and geometry.
   * @param material Material of the whole space
   * @param type P or SV
   * @param take_off_deg Take-off angle from +z, in degrees
   * @param azimuth_deg Azimuth from +x toward +y, in degrees
   * @param origin Point \f$ x_0 \f$ the wavefront crosses at the origin time
   * @param origin_time Time \f$ t_0 \f$ the wavefront crosses the origin
   * @param amplitude Amplitude \f$ A \f$
   * @param waveform Displacement waveform \f$ f \f$
   */
  PlaneWave(const Material &material, const WaveType type,
            const double take_off_deg, const double azimuth_deg,
            const Vector &origin, const double origin_time,
            const double amplitude, const SourceTimeFunction &waveform)
      : material_(material), origin_(origin), origin_time_(origin_time),
        amplitude_(amplitude), waveform_(waveform),
        speed_(type == WaveType::p ? material.vp : material.vs) {
    const double degree = std::numbers::pi / 180;
    const double theta = take_off_deg * degree;
    const double phi = azimuth_deg * degree;
    direction_ = { std::sin(theta) * std::cos(phi),
                   std::sin(theta) * std::sin(phi), std::cos(theta) };
    polarization_ =
        type == WaveType::p
            ? direction_
            : Vector{ std::cos(theta) * std::cos(phi),
                      std::cos(theta) * std::sin(phi), -std::sin(theta) };
  }

  /** @brief Propagation direction. @return \f$ \hat{k} \f$ */
  const Vector &direction() const { return direction_; }

  /** @brief Polarization. @return \f$ \hat{d} \f$ */
  const Vector &polarization() const { return polarization_; }

  /** @brief Propagation speed. @return \f$ c \f$ */
  double speed() const { return speed_; }

  /**
   * @brief Displacement.
   * @param x Position
   * @param t Time
   * @return \f$ A\,\hat{d}\,f(\tau) \f$
   */
  Vector displacement(const Vector &x, const double t) const {
    return scaled_polarization(waveform_.derivative(tau(x, t), 0));
  }

  /**
   * @brief Velocity.
   * @param x Position
   * @param t Time
   * @return \f$ A\,\hat{d}\,f'(\tau) \f$
   */
  Vector velocity(const Vector &x, const double t) const {
    return scaled_polarization(waveform_.derivative(tau(x, t), 1));
  }

  /**
   * @brief Acceleration.
   * @param x Position
   * @param t Time
   * @return \f$ A\,\hat{d}\,f''(\tau) \f$
   */
  Vector acceleration(const Vector &x, const double t) const {
    return scaled_polarization(waveform_.derivative(tau(x, t), 2));
  }

  /**
   * @brief Strain.
   *
   * \f$ \partial_j u_i = -(A/c)\,f'(\tau)\,\hat{d}_i \hat{k}_j \f$, so
   * \f$ \varepsilon = -(A/c)\,f'(\tau)\,\mathrm{sym}(\hat{d} \otimes \hat{k})
   * \f$.
   *
   * @param x Position
   * @param t Time
   * @return Strain tensor
   */
  Tensor strain(const Vector &x, const double t) const {
    const double factor =
        -amplitude_ / speed_ * waveform_.derivative(tau(x, t), 1);
    Tensor epsilon;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        epsilon[i][j] = factor *
                        (polarization_[i] * direction_[j] +
                         polarization_[j] * direction_[i]) /
                        2;
    return epsilon;
  }

  /**
   * @brief Stress.
   * @param x Position
   * @param t Time
   * @return \f$ \sigma = \lambda\,\mathrm{tr}(\varepsilon)\,I +
   * 2\mu\,\varepsilon \f$
   */
  Tensor stress(const Vector &x, const double t) const {
    const Tensor epsilon = strain(x, t);
    const double trace = epsilon[0][0] + epsilon[1][1] + epsilon[2][2];
    Tensor sigma;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        sigma[i][j] = 2 * material_.mu() * epsilon[i][j] +
                      (i == j ? material_.lambda() * trace : 0);
    return sigma;
  }

  /**
   * @brief Traction on a surface.
   * @param x Position
   * @param normal Unit normal of the surface
   * @param t Time
   * @return \f$ \sigma \hat{n} \f$
   */
  Vector traction(const Vector &x, const Vector &normal, const double t) const {
    const Tensor sigma = stress(x, t);
    Vector traction;
    for (int i = 0; i < 3; ++i)
      traction[i] = sigma[i][0] * normal[0] + sigma[i][1] * normal[1] +
                    sigma[i][2] * normal[2];
    return traction;
  }

private:
  double tau(const Vector &x, const double t) const {
    double distance = 0;
    for (int i = 0; i < 3; ++i)
      distance += direction_[i] * (x[i] - origin_[i]);
    return t - origin_time_ - distance / speed_;
  }

  Vector scaled_polarization(const double value) const {
    return { amplitude_ * polarization_[0] * value,
             amplitude_ * polarization_[1] * value,
             amplitude_ * polarization_[2] * value };
  }

  Material material_;
  Vector origin_;
  double origin_time_;
  double amplitude_;
  SourceTimeFunction waveform_;
  double speed_;
  Vector direction_{};
  Vector polarization_{};
};

} // namespace plane_wave
