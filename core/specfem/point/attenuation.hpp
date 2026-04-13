#pragma once

#include "specfem/constants.hpp"
#include "specfem/data_access/accessor.hpp"
#include "specfem/element.hpp"
#include "specfem/element/attributes.hpp"
#include "specfem/utilities/is_close.hpp"
#include <sstream>

namespace specfem::point {

/**
 * @brief Combined attenuation struct for viscoelastic simulations.
 *
 * Merges `attenuation_factors` and `memory_variable` into a single type that
 * carries both the constant-Q relaxation coefficients and the per-SLS memory
 * variable arrays. Using one struct eliminates the need to pass the two
 * separate objects everywhere they are consumed together.
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 * @tparam MediumTag Medium type (elastic)
 * @tparam AttenuationTag Attenuation model (constant_isotropic)
 * @tparam UseSIMD Enable SIMD vectorization
 */
template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::element::attenuation_tag AttenuationTag, bool UseSIMD>
struct attenuation;

/**
 * @brief Template specialization for no attenuation.
 *
 * Empty type — carries no data and requires no computation. Used when
 * attenuation is disabled so that downstream code can be written generically
 * against the attenuation interface without runtime cost.
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 * @tparam MediumTag    Medium type
 * @tparam UseSIMD      Enable SIMD vectorization
 */
template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag, bool UseSIMD>
struct attenuation<DimensionTag, MediumTag,
                   specfem::element::attenuation_tag::none, UseSIMD>
    : public specfem::data_access::Accessor<
          specfem::datatype::AccessorType::point,
          specfem::data_access::DataClassType::attenuation, DimensionTag,
          UseSIMD> {
private:
  using base_type = specfem::data_access::Accessor<
      specfem::datatype::AccessorType::point,
      specfem::data_access::DataClassType::attenuation, DimensionTag, UseSIMD>;

public:
  // -----------------------------------------------------------------------
  // Static properties
  // -----------------------------------------------------------------------
  constexpr static specfem::element::dimension_tag dimension_tag = DimensionTag;
  constexpr static specfem::element::attenuation_tag attenuation_tag =
      specfem::element::attenuation_tag::none;
  constexpr static int N_SLS = specfem::constants::N_SLS;
  constexpr static bool using_simd = UseSIMD;

  // -----------------------------------------------------------------------
  // Type aliases
  // -----------------------------------------------------------------------
  using simd = typename base_type::template simd<type_real>;
  using value_type = typename base_type::template vector_type<type_real, N_SLS>;

  // -----------------------------------------------------------------------
  // Constructors
  // -----------------------------------------------------------------------

  /** @brief Default constructor — no-op. */
  KOKKOS_FUNCTION
  attenuation() = default;
};

// ---------------------------------------------------------------------------
// dim2 specialization
// ---------------------------------------------------------------------------

/**
 * @brief Attenuation state for 2D elastic medium with constant isotropic Q.
 *
 * Stores the Runge-Kutta integration factors together with the three
 * independent 2D memory variable components (Rxx, Rxz, Rkappa).
 *
 * @tparam UseSIMD Enable SIMD vectorization
 */
template <bool UseSIMD>
struct attenuation<specfem::element::dimension_tag::dim2,
                   specfem::element::medium_tag::elastic_psv,
                   specfem::element::attenuation_tag::constant_isotropic,
                   UseSIMD>
    : public specfem::data_access::Accessor<
          specfem::datatype::AccessorType::point,
          specfem::data_access::DataClassType::attenuation,
          specfem::element::dimension_tag::dim2, UseSIMD> {
private:
  using base_type = specfem::data_access::Accessor<
      specfem::datatype::AccessorType::point,
      specfem::data_access::DataClassType::attenuation,
      specfem::element::dimension_tag::dim2, UseSIMD>;

public:
  // -----------------------------------------------------------------------
  // Static properties
  // -----------------------------------------------------------------------
  constexpr static specfem::element::dimension_tag dimension_tag =
      specfem::element::dimension_tag::dim2;
  constexpr static specfem::element::medium_tag medium_tag =
      specfem::element::medium_tag::elastic_psv;
  constexpr static specfem::element::property_tag property_tag =
      specfem::element::property_tag::isotropic;
  constexpr static specfem::element::attenuation_tag attenuation_tag =
      specfem::element::attenuation_tag::constant_isotropic;
  constexpr static int N_SLS = specfem::constants::N_SLS;
  constexpr static bool using_simd = UseSIMD;

  // -----------------------------------------------------------------------
  // Type aliases
  // -----------------------------------------------------------------------
  using simd = typename base_type::template simd<type_real>;
  using value_type = typename base_type::template vector_type<type_real, N_SLS>;

  // -----------------------------------------------------------------------
  // Data members — attenuation factors
  // -----------------------------------------------------------------------
  value_type kappa_relaxation_rate;
  value_type mu_relaxation_rate;
  /// Runge-Kutta coefficients; one entry per SLS mechanism (computed from
  /// tau_sigma and deltat; constant throughout a run).
  value_type alpha_rk;
  value_type beta_rk;
  value_type gamma_rk;

  // -----------------------------------------------------------------------
  // Data members — memory variables (dim2)
  // -----------------------------------------------------------------------
  value_type Rxx;
  value_type Rxz;
  value_type Rkappa;

  // -----------------------------------------------------------------------
  // Compile-time constants for du tensor
  // -----------------------------------------------------------------------
  static constexpr int components = specfem::element::attributes<
      specfem::element::dimension_tag::dim2,
      specfem::element::medium_tag::elastic_psv>::components;
  static constexpr int num_dimensions = specfem::element::attributes<
      specfem::element::dimension_tag::dim2,
      specfem::element::medium_tag::elastic_psv>::dimension;

  // -----------------------------------------------------------------------
  // Data members — du_att tensor (Taylor step: grad(u + dt*v))
  // -----------------------------------------------------------------------
  /// du_att = grad(u + dt*v) from previous time step (Sn for SLS update)
  typename base_type::template tensor_type<type_real, components,
                                           num_dimensions>
      du;

  // -----------------------------------------------------------------------
  // Constructors
  // -----------------------------------------------------------------------

  /**
   * @brief Default constructor — zeroes all R fields; factor fields default.
   */
  KOKKOS_FUNCTION
  attenuation() { this->init(); }

  /**
   * @brief Full value constructor.
   *
   * @param kappa_relaxation_rate Scaling factor for kappa attenuation
   * @param mu_relaxation_rate    Scaling factor for mu attenuation
   * @param alpha_rk            Runge-Kutta alpha factors (per SLS)
   * @param beta_rk             Runge-Kutta beta factors (per SLS)
   * @param gamma_rk            Runge-Kutta gamma factors (per SLS)
   * @param Rxx                 Memory variable R_xx
   * @param Rxz                 Memory variable R_xz
   * @param Rkappa              Memory variable R_kappa
   * @param du                  Gradient of Taylor-predicted displacement
   */
  KOKKOS_FUNCTION
  attenuation(const value_type &kappa_relaxation_rate,
              const value_type &mu_relaxation_rate, const value_type &alpha_rk,
              const value_type &beta_rk, const value_type &gamma_rk,
              const value_type &Rxx, const value_type &Rxz,
              const value_type &Rkappa,
              const typename base_type::template tensor_type<
                  type_real, components, num_dimensions> &du)
      : kappa_relaxation_rate(kappa_relaxation_rate),
        mu_relaxation_rate(mu_relaxation_rate), alpha_rk(alpha_rk),
        beta_rk(beta_rk), gamma_rk(gamma_rk), Rxx(Rxx), Rxz(Rxz),
        Rkappa(Rkappa), du(du) {}

  // -----------------------------------------------------------------------
  // Methods
  // -----------------------------------------------------------------------

  /** @brief Zero all memory variable (R) fields and du. */
  KOKKOS_FUNCTION
  void init() {
    this->Rxx = value_type(typename value_type::value_type(0));
    this->Rxz = value_type(typename value_type::value_type(0));
    this->Rkappa = value_type(typename value_type::value_type(0));
    for (int ic = 0; ic < components; ++ic)
      for (int id = 0; id < num_dimensions; ++id)
        this->du[ic][id] = typename simd::datatype(0);
  }

  /** @brief Component-wise addition on R fields and du; RK/common factors
   * copied. */
  KOKKOS_FUNCTION attenuation operator+(const attenuation &rhs) const {
    attenuation result;
    result.kappa_relaxation_rate = this->kappa_relaxation_rate;
    result.mu_relaxation_rate = this->mu_relaxation_rate;
    result.alpha_rk = this->alpha_rk;
    result.beta_rk = this->beta_rk;
    result.gamma_rk = this->gamma_rk;
    constexpr int N = specfem::constants::N_SLS;
    for (int i = 0; i < N; ++i) {
      result.Rxx(i) = this->Rxx(i) + rhs.Rxx(i);
      result.Rxz(i) = this->Rxz(i) + rhs.Rxz(i);
      result.Rkappa(i) = this->Rkappa(i) + rhs.Rkappa(i);
    }
    for (int ic = 0; ic < components; ++ic)
      for (int id = 0; id < num_dimensions; ++id)
        result.du[ic][id] = this->du[ic][id] + rhs.du[ic][id];
    return result;
  }

  /** @brief In-place addition on R fields and du. */
  KOKKOS_FUNCTION attenuation &operator+=(const attenuation &rhs) {
    constexpr int N = specfem::constants::N_SLS;
    for (int i = 0; i < N; ++i) {
      this->Rxx(i) += rhs.Rxx(i);
      this->Rxz(i) += rhs.Rxz(i);
      this->Rkappa(i) += rhs.Rkappa(i);
    }
    for (int ic = 0; ic < components; ++ic)
      for (int id = 0; id < num_dimensions; ++id)
        this->du[ic][id] += rhs.du[ic][id];
    return *this;
  }

  /** @brief Scalar multiplication on R fields and du; RK/common factors copied.
   */
  KOKKOS_FUNCTION attenuation operator*(const type_real &rhs) const {
    attenuation result;
    result.kappa_relaxation_rate = this->kappa_relaxation_rate;
    result.mu_relaxation_rate = this->mu_relaxation_rate;
    result.alpha_rk = this->alpha_rk;
    result.beta_rk = this->beta_rk;
    result.gamma_rk = this->gamma_rk;
    constexpr int N = specfem::constants::N_SLS;
    for (int i = 0; i < N; ++i) {
      result.Rxx(i) = this->Rxx(i) * rhs;
      result.Rxz(i) = this->Rxz(i) * rhs;
      result.Rkappa(i) = this->Rkappa(i) * rhs;
    }
    for (int ic = 0; ic < components; ++ic)
      for (int id = 0; id < num_dimensions; ++id)
        result.du[ic][id] = this->du[ic][id] * rhs;
    return result;
  }

  /** @brief Equality — compares all fields (factors, R, and du). */
  KOKKOS_INLINE_FUNCTION
  bool operator==(const attenuation &other) const {
    constexpr int N = specfem::constants::N_SLS;
    if (!(kappa_relaxation_rate == other.kappa_relaxation_rate))
      return false;
    if (!(mu_relaxation_rate == other.mu_relaxation_rate))
      return false;
    for (int i = 0; i < N; ++i) {
      if (!specfem::utilities::is_close(alpha_rk(i), other.alpha_rk(i)))
        return false;
      if (!specfem::utilities::is_close(beta_rk(i), other.beta_rk(i)))
        return false;
      if (!specfem::utilities::is_close(gamma_rk(i), other.gamma_rk(i)))
        return false;
      if (!specfem::utilities::is_close(Rxx(i), other.Rxx(i)))
        return false;
      if (!specfem::utilities::is_close(Rxz(i), other.Rxz(i)))
        return false;
      if (!specfem::utilities::is_close(Rkappa(i), other.Rkappa(i)))
        return false;
    }
    for (int ic = 0; ic < components; ++ic)
      for (int id = 0; id < num_dimensions; ++id)
        if (!specfem::utilities::is_close(du[ic][id], other.du[ic][id]))
          return false;
    return true;
  }

  /** @brief String representation (non-SIMD only). */
  std::string print() const {
    std::ostringstream oss;
    oss << "Attenuation Factors:\n";
    for (int i = 0; i < N_SLS; ++i)
      oss << "  kappa_relaxation_rate(" << i
          << ") = " << kappa_relaxation_rate(i) << "\n";
    for (int i = 0; i < N_SLS; ++i)
      oss << "  mu_relaxation_rate(" << i << ") = " << mu_relaxation_rate(i)
          << "\n";
    for (int i = 0; i < N_SLS; ++i)
      oss << "  alpha_rk(" << i << ") = " << alpha_rk(i) << "\n";
    for (int i = 0; i < N_SLS; ++i)
      oss << "  beta_rk(" << i << ") = " << beta_rk(i) << "\n";
    for (int i = 0; i < N_SLS; ++i)
      oss << "  gamma_rk(" << i << ") = " << gamma_rk(i) << "\n";
    oss << "Memory Variables (dim2):\n";
    for (int i = 0; i < N_SLS; ++i)
      oss << "  Rxx(" << i << ") = " << Rxx(i) << "\n";
    for (int i = 0; i < N_SLS; ++i)
      oss << "  Rxz(" << i << ") = " << Rxz(i) << "\n";
    for (int i = 0; i < N_SLS; ++i)
      oss << "  Rkappa(" << i << ") = " << Rkappa(i) << "\n";
    oss << "Taylor Step du (dim2):\n";
    for (int ic = 0; ic < components; ++ic)
      for (int id = 0; id < num_dimensions; ++id)
        oss << "  du[" << ic << "][" << id << "] = " << du[ic][id] << "\n";
    return oss.str();
  }
};

// ---------------------------------------------------------------------------
// dim3 specialization
// ---------------------------------------------------------------------------

/**
 * @brief Attenuation state for 3D elastic medium with constant isotropic Q.
 *
 * Stores the Runge-Kutta integration factors together with the six
 * independent 3D memory variable components (Rxx, Ryy, Rxy, Rxz, Ryz,
 * Rkappa).
 *
 * @tparam UseSIMD Enable SIMD vectorization
 */
template <bool UseSIMD>
struct attenuation<specfem::element::dimension_tag::dim3,
                   specfem::element::medium_tag::elastic,
                   specfem::element::attenuation_tag::constant_isotropic,
                   UseSIMD>
    : public specfem::data_access::Accessor<
          specfem::datatype::AccessorType::point,
          specfem::data_access::DataClassType::attenuation,
          specfem::element::dimension_tag::dim3, UseSIMD> {
private:
  using base_type = specfem::data_access::Accessor<
      specfem::datatype::AccessorType::point,
      specfem::data_access::DataClassType::attenuation,
      specfem::element::dimension_tag::dim3, UseSIMD>;

public:
  // -----------------------------------------------------------------------
  // Static properties
  // -----------------------------------------------------------------------
  constexpr static specfem::element::dimension_tag dimension_tag =
      specfem::element::dimension_tag::dim3;
  constexpr static specfem::element::medium_tag medium_tag =
      specfem::element::medium_tag::elastic;
  constexpr static specfem::element::property_tag property_tag =
      specfem::element::property_tag::isotropic;
  constexpr static specfem::element::attenuation_tag attenuation_tag =
      specfem::element::attenuation_tag::constant_isotropic;
  constexpr static int N_SLS = specfem::constants::N_SLS;
  constexpr static bool using_simd = UseSIMD;

  // -----------------------------------------------------------------------
  // Type aliases
  // -----------------------------------------------------------------------
  using simd = typename base_type::template simd<type_real>;
  using value_type = typename base_type::template vector_type<type_real, N_SLS>;

  // -----------------------------------------------------------------------
  // Data members — attenuation factors
  // -----------------------------------------------------------------------
  value_type kappa_relaxation_rate;
  value_type mu_relaxation_rate;
  /// Runge-Kutta coefficients; one entry per SLS mechanism (computed from
  /// tau_sigma and deltat; constant throughout a run).
  value_type alpha_rk;
  value_type beta_rk;
  value_type gamma_rk;

  // -----------------------------------------------------------------------
  // Data members — memory variables (dim3)
  // -----------------------------------------------------------------------
  value_type Rxx;
  value_type Ryy;
  value_type Rxy;
  value_type Rxz;
  value_type Ryz;
  value_type Rkappa;

  // -----------------------------------------------------------------------
  // Compile-time constants for du tensor
  // -----------------------------------------------------------------------
  static constexpr int components = specfem::element::attributes<
      specfem::element::dimension_tag::dim3,
      specfem::element::medium_tag::elastic>::components;
  static constexpr int num_dimensions = specfem::element::attributes<
      specfem::element::dimension_tag::dim3,
      specfem::element::medium_tag::elastic>::dimension;

  // -----------------------------------------------------------------------
  // Data members — du_att tensor (Taylor step: grad(u + dt*v))
  // -----------------------------------------------------------------------
  /// du_att = grad(u + dt*v) from previous time step (Sn for SLS update)
  typename base_type::template tensor_type<type_real, components,
                                           num_dimensions>
      du;

  // -----------------------------------------------------------------------
  // Constructors
  // -----------------------------------------------------------------------

  /**
   * @brief Default constructor — zeroes all R fields; factor fields default.
   */
  KOKKOS_FUNCTION
  attenuation() { this->init(); }

  /**
   * @brief Full value constructor.
   *
   * @param kappa_relaxation_rate Scaling factor for kappa attenuation
   * @param mu_relaxation_rate    Scaling factor for mu attenuation
   * @param alpha_rk            Runge-Kutta alpha factors (per SLS)
   * @param beta_rk             Runge-Kutta beta factors (per SLS)
   * @param gamma_rk            Runge-Kutta gamma factors (per SLS)
   * @param Rxx                 Memory variable R_xx
   * @param Ryy                 Memory variable R_yy
   * @param Rxy                 Memory variable R_xy
   * @param Rxz                 Memory variable R_xz
   * @param Ryz                 Memory variable R_yz
   * @param Rkappa              Memory variable R_kappa
   * @param du                  Gradient of Taylor-predicted displacement
   */
  KOKKOS_FUNCTION
  attenuation(const value_type &kappa_relaxation_rate,
              const value_type &mu_relaxation_rate, const value_type &alpha_rk,
              const value_type &beta_rk, const value_type &gamma_rk,
              const value_type &Rxx, const value_type &Ryy,
              const value_type &Rxy, const value_type &Rxz,
              const value_type &Ryz, const value_type &Rkappa,
              const typename base_type::template tensor_type<
                  type_real, components, num_dimensions> &du)
      : kappa_relaxation_rate(kappa_relaxation_rate),
        mu_relaxation_rate(mu_relaxation_rate), alpha_rk(alpha_rk),
        beta_rk(beta_rk), gamma_rk(gamma_rk), Rxx(Rxx), Ryy(Ryy), Rxy(Rxy),
        Rxz(Rxz), Ryz(Ryz), Rkappa(Rkappa), du(du) {}

  // -----------------------------------------------------------------------
  // Methods
  // -----------------------------------------------------------------------

  /** @brief Zero all memory variable (R) fields and du. */
  KOKKOS_FUNCTION
  void init() {
    this->Rxx = value_type(typename value_type::value_type(0));
    this->Ryy = value_type(typename value_type::value_type(0));
    this->Rxy = value_type(typename value_type::value_type(0));
    this->Rxz = value_type(typename value_type::value_type(0));
    this->Ryz = value_type(typename value_type::value_type(0));
    this->Rkappa = value_type(typename value_type::value_type(0));
    for (int ic = 0; ic < components; ++ic)
      for (int id = 0; id < num_dimensions; ++id)
        this->du[ic][id] = typename simd::datatype(0);
  }

  /** @brief Component-wise addition on R fields and du; RK/common factors
   * copied. */
  KOKKOS_FUNCTION attenuation operator+(const attenuation &rhs) const {
    attenuation result;
    result.kappa_relaxation_rate = this->kappa_relaxation_rate;
    result.mu_relaxation_rate = this->mu_relaxation_rate;
    result.alpha_rk = this->alpha_rk;
    result.beta_rk = this->beta_rk;
    result.gamma_rk = this->gamma_rk;
    constexpr int N = specfem::constants::N_SLS;
    for (int i = 0; i < N; ++i) {
      result.Rxx(i) = this->Rxx(i) + rhs.Rxx(i);
      result.Ryy(i) = this->Ryy(i) + rhs.Ryy(i);
      result.Rxy(i) = this->Rxy(i) + rhs.Rxy(i);
      result.Rxz(i) = this->Rxz(i) + rhs.Rxz(i);
      result.Ryz(i) = this->Ryz(i) + rhs.Ryz(i);
      result.Rkappa(i) = this->Rkappa(i) + rhs.Rkappa(i);
    }
    for (int ic = 0; ic < components; ++ic)
      for (int id = 0; id < num_dimensions; ++id)
        result.du[ic][id] = this->du[ic][id] + rhs.du[ic][id];
    return result;
  }

  /** @brief In-place addition on R fields and du. */
  KOKKOS_FUNCTION attenuation &operator+=(const attenuation &rhs) {
    constexpr int N = specfem::constants::N_SLS;
    for (int i = 0; i < N; ++i) {
      this->Rxx(i) += rhs.Rxx(i);
      this->Ryy(i) += rhs.Ryy(i);
      this->Rxy(i) += rhs.Rxy(i);
      this->Rxz(i) += rhs.Rxz(i);
      this->Ryz(i) += rhs.Ryz(i);
      this->Rkappa(i) += rhs.Rkappa(i);
    }
    for (int ic = 0; ic < components; ++ic)
      for (int id = 0; id < num_dimensions; ++id)
        this->du[ic][id] += rhs.du[ic][id];
    return *this;
  }

  /** @brief Scalar multiplication on R fields and du; RK/common factors copied.
   */
  KOKKOS_FUNCTION attenuation operator*(const type_real &rhs) const {
    attenuation result;
    result.kappa_relaxation_rate = this->kappa_relaxation_rate;
    result.mu_relaxation_rate = this->mu_relaxation_rate;
    result.alpha_rk = this->alpha_rk;
    result.beta_rk = this->beta_rk;
    result.gamma_rk = this->gamma_rk;
    constexpr int N = specfem::constants::N_SLS;
    for (int i = 0; i < N; ++i) {
      result.Rxx(i) = this->Rxx(i) * rhs;
      result.Ryy(i) = this->Ryy(i) * rhs;
      result.Rxy(i) = this->Rxy(i) * rhs;
      result.Rxz(i) = this->Rxz(i) * rhs;
      result.Ryz(i) = this->Ryz(i) * rhs;
      result.Rkappa(i) = this->Rkappa(i) * rhs;
    }
    for (int ic = 0; ic < components; ++ic)
      for (int id = 0; id < num_dimensions; ++id)
        result.du[ic][id] = this->du[ic][id] * rhs;
    return result;
  }

  /** @brief Equality — compares all fields (factors, R, and du). */
  KOKKOS_INLINE_FUNCTION
  bool operator==(const attenuation &other) const {
    constexpr int N = specfem::constants::N_SLS;
    if (!(kappa_relaxation_rate == other.kappa_relaxation_rate))
      return false;
    if (!(mu_relaxation_rate == other.mu_relaxation_rate))
      return false;
    for (int i = 0; i < N; ++i) {
      if (!specfem::utilities::is_close(alpha_rk(i), other.alpha_rk(i)))
        return false;
      if (!specfem::utilities::is_close(beta_rk(i), other.beta_rk(i)))
        return false;
      if (!specfem::utilities::is_close(gamma_rk(i), other.gamma_rk(i)))
        return false;
      if (!specfem::utilities::is_close(Rxx(i), other.Rxx(i)))
        return false;
      if (!specfem::utilities::is_close(Ryy(i), other.Ryy(i)))
        return false;
      if (!specfem::utilities::is_close(Rxy(i), other.Rxy(i)))
        return false;
      if (!specfem::utilities::is_close(Rxz(i), other.Rxz(i)))
        return false;
      if (!specfem::utilities::is_close(Ryz(i), other.Ryz(i)))
        return false;
      if (!specfem::utilities::is_close(Rkappa(i), other.Rkappa(i)))
        return false;
    }
    for (int ic = 0; ic < components; ++ic)
      for (int id = 0; id < num_dimensions; ++id)
        if (!specfem::utilities::is_close(du[ic][id], other.du[ic][id]))
          return false;
    return true;
  }

  /** @brief String representation (non-SIMD only). */
  std::string print() const {
    std::ostringstream oss;
    oss << "Attenuation Factors:\n";
    for (int i = 0; i < N_SLS; ++i)
      oss << "  kappa_relaxation_rate(" << i
          << ") = " << kappa_relaxation_rate(i) << "\n";
    for (int i = 0; i < N_SLS; ++i)
      oss << "  mu_relaxation_rate(" << i << ") = " << mu_relaxation_rate(i)
          << "\n";
    for (int i = 0; i < N_SLS; ++i)
      oss << "  alpha_rk(" << i << ") = " << alpha_rk(i) << "\n";
    for (int i = 0; i < N_SLS; ++i)
      oss << "  beta_rk(" << i << ") = " << beta_rk(i) << "\n";
    for (int i = 0; i < N_SLS; ++i)
      oss << "  gamma_rk(" << i << ") = " << gamma_rk(i) << "\n";
    oss << "Memory Variables (dim3):\n";
    for (int i = 0; i < N_SLS; ++i)
      oss << "  Rxx(" << i << ") = " << Rxx(i) << "\n";
    for (int i = 0; i < N_SLS; ++i)
      oss << "  Ryy(" << i << ") = " << Ryy(i) << "\n";
    for (int i = 0; i < N_SLS; ++i)
      oss << "  Rxy(" << i << ") = " << Rxy(i) << "\n";
    for (int i = 0; i < N_SLS; ++i)
      oss << "  Rxz(" << i << ") = " << Rxz(i) << "\n";
    for (int i = 0; i < N_SLS; ++i)
      oss << "  Ryz(" << i << ") = " << Ryz(i) << "\n";
    for (int i = 0; i < N_SLS; ++i)
      oss << "  Rkappa(" << i << ") = " << Rkappa(i) << "\n";
    oss << "Taylor Step du (dim3):\n";
    for (int ic = 0; ic < components; ++ic)
      for (int id = 0; id < num_dimensions; ++id)
        oss << "  du[" << ic << "][" << id << "] = " << du[ic][id] << "\n";
    return oss.str();
  }
};

} // namespace specfem::point
