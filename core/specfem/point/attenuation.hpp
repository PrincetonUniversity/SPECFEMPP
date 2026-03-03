#pragma once

#include "specfem/constants.hpp"
#include "specfem/data_access/accessor.hpp"
#include "specfem/element.hpp"
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
  constexpr static specfem::element::attenuation_tag attenuation_tag =
      specfem::element::attenuation_tag::constant_isotropic;
  constexpr static int N_SLS = specfem::constants::N_SLS;
  constexpr static bool using_simd = UseSIMD;

  // -----------------------------------------------------------------------
  // Type aliases
  // -----------------------------------------------------------------------
  using simd = typename base_type::template simd<type_real>;
  using common_factor_type =
      typename base_type::template vector_type<type_real, N_SLS>;
  using value_type = typename base_type::template vector_type<type_real, N_SLS>;

  // -----------------------------------------------------------------------
  // Data members — attenuation factors
  // -----------------------------------------------------------------------
  common_factor_type kappa_common_factor;
  common_factor_type mu_common_factor;
  type_real alpha_rk;
  type_real beta_rk;
  type_real gamma_rk;

  // -----------------------------------------------------------------------
  // Data members — memory variables (dim2)
  // -----------------------------------------------------------------------
  value_type Rxx;
  value_type Rxz;
  value_type Rkappa;

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
   * @param kappa_common_factor Common factor for kappa attenuation
   * @param mu_common_factor    Common factor for mu attenuation
   * @param alpha_rk            Runge-Kutta alpha factor
   * @param beta_rk             Runge-Kutta beta factor
   * @param gamma_rk            Runge-Kutta gamma factor
   * @param Rxx                 Memory variable R_xx
   * @param Rxz                 Memory variable R_xz
   * @param Rkappa              Memory variable R_kappa
   */
  KOKKOS_FUNCTION
  attenuation(const common_factor_type &kappa_common_factor,
              const common_factor_type &mu_common_factor,
              const type_real &alpha_rk, const type_real &beta_rk,
              const type_real &gamma_rk, const value_type &Rxx,
              const value_type &Rxz, const value_type &Rkappa)
      : kappa_common_factor(kappa_common_factor),
        mu_common_factor(mu_common_factor), alpha_rk(alpha_rk),
        beta_rk(beta_rk), gamma_rk(gamma_rk), Rxx(Rxx), Rxz(Rxz),
        Rkappa(Rkappa) {}

  // -----------------------------------------------------------------------
  // Methods
  // -----------------------------------------------------------------------

  /** @brief Zero all memory variable (R) fields. */
  KOKKOS_FUNCTION
  void init() {
    this->Rxx = value_type(typename value_type::value_type(0));
    this->Rxz = value_type(typename value_type::value_type(0));
    this->Rkappa = value_type(typename value_type::value_type(0));
  }

  /** @brief Component-wise addition on R fields. */
  KOKKOS_FUNCTION attenuation operator+(const attenuation &rhs) const {
    attenuation result;
    result.kappa_common_factor = this->kappa_common_factor;
    result.mu_common_factor = this->mu_common_factor;
    result.alpha_rk = this->alpha_rk;
    result.beta_rk = this->beta_rk;
    result.gamma_rk = this->gamma_rk;
    constexpr int N = specfem::constants::N_SLS;
    for (int i = 0; i < N; ++i) {
      result.Rxx(i) = this->Rxx(i) + rhs.Rxx(i);
      result.Rxz(i) = this->Rxz(i) + rhs.Rxz(i);
      result.Rkappa(i) = this->Rkappa(i) + rhs.Rkappa(i);
    }
    return result;
  }

  /** @brief In-place addition on R fields. */
  KOKKOS_FUNCTION attenuation &operator+=(const attenuation &rhs) {
    constexpr int N = specfem::constants::N_SLS;
    for (int i = 0; i < N; ++i) {
      this->Rxx(i) += rhs.Rxx(i);
      this->Rxz(i) += rhs.Rxz(i);
      this->Rkappa(i) += rhs.Rkappa(i);
    }
    return *this;
  }

  /** @brief Scalar multiplication on R fields. */
  KOKKOS_FUNCTION attenuation operator*(const type_real &rhs) const {
    attenuation result;
    result.kappa_common_factor = this->kappa_common_factor;
    result.mu_common_factor = this->mu_common_factor;
    result.alpha_rk = this->alpha_rk;
    result.beta_rk = this->beta_rk;
    result.gamma_rk = this->gamma_rk;
    constexpr int N = specfem::constants::N_SLS;
    for (int i = 0; i < N; ++i) {
      result.Rxx(i) = this->Rxx(i) * rhs;
      result.Rxz(i) = this->Rxz(i) * rhs;
      result.Rkappa(i) = this->Rkappa(i) * rhs;
    }
    return result;
  }

  /** @brief Equality — compares all fields (factors and R). */
  KOKKOS_INLINE_FUNCTION
  bool operator==(const attenuation &other) const {
    return kappa_common_factor == other.kappa_common_factor &&
           mu_common_factor == other.mu_common_factor &&
           alpha_rk == other.alpha_rk && beta_rk == other.beta_rk &&
           gamma_rk == other.gamma_rk && Rxx == other.Rxx && Rxz == other.Rxz &&
           Rkappa == other.Rkappa;
  }

  /** @brief String representation (non-SIMD only). */
  std::string print() const {
    std::ostringstream oss;
    oss << "Attenuation Factors:\n";
    for (int i = 0; i < N_SLS; ++i)
      oss << "  kappa_common_factor(" << i << ") = " << kappa_common_factor(i)
          << "\n";
    for (int i = 0; i < N_SLS; ++i)
      oss << "  mu_common_factor(" << i << ") = " << mu_common_factor(i)
          << "\n";
    oss << "  alpha_rk = " << alpha_rk << "\n";
    oss << "  beta_rk = " << beta_rk << "\n";
    oss << "  gamma_rk = " << gamma_rk << "\n";
    oss << "Memory Variables (dim2):\n";
    for (int i = 0; i < N_SLS; ++i)
      oss << "  Rxx(" << i << ") = " << Rxx(i) << "\n";
    for (int i = 0; i < N_SLS; ++i)
      oss << "  Rxz(" << i << ") = " << Rxz(i) << "\n";
    for (int i = 0; i < N_SLS; ++i)
      oss << "  Rkappa(" << i << ") = " << Rkappa(i) << "\n";
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
  constexpr static specfem::element::attenuation_tag attenuation_tag =
      specfem::element::attenuation_tag::constant_isotropic;
  constexpr static int N_SLS = specfem::constants::N_SLS;
  constexpr static bool using_simd = UseSIMD;

  // -----------------------------------------------------------------------
  // Type aliases
  // -----------------------------------------------------------------------
  using simd = typename base_type::template simd<type_real>;
  using common_factor_type =
      typename base_type::template vector_type<type_real, N_SLS>;
  using value_type = typename base_type::template vector_type<type_real, N_SLS>;

  // -----------------------------------------------------------------------
  // Data members — attenuation factors
  // -----------------------------------------------------------------------
  common_factor_type kappa_common_factor;
  common_factor_type mu_common_factor;
  type_real alpha_rk;
  type_real beta_rk;
  type_real gamma_rk;

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
   * @param kappa_common_factor Common factor for kappa attenuation
   * @param mu_common_factor    Common factor for mu attenuation
   * @param alpha_rk            Runge-Kutta alpha factor
   * @param beta_rk             Runge-Kutta beta factor
   * @param gamma_rk            Runge-Kutta gamma factor
   * @param Rxx                 Memory variable R_xx
   * @param Ryy                 Memory variable R_yy
   * @param Rxy                 Memory variable R_xy
   * @param Rxz                 Memory variable R_xz
   * @param Ryz                 Memory variable R_yz
   * @param Rkappa              Memory variable R_kappa
   */
  KOKKOS_FUNCTION
  attenuation(const common_factor_type &kappa_common_factor,
              const common_factor_type &mu_common_factor,
              const type_real &alpha_rk, const type_real &beta_rk,
              const type_real &gamma_rk, const value_type &Rxx,
              const value_type &Ryy, const value_type &Rxy,
              const value_type &Rxz, const value_type &Ryz,
              const value_type &Rkappa)
      : kappa_common_factor(kappa_common_factor),
        mu_common_factor(mu_common_factor), alpha_rk(alpha_rk),
        beta_rk(beta_rk), gamma_rk(gamma_rk), Rxx(Rxx), Ryy(Ryy), Rxy(Rxy),
        Rxz(Rxz), Ryz(Ryz), Rkappa(Rkappa) {}

  // -----------------------------------------------------------------------
  // Methods
  // -----------------------------------------------------------------------

  /** @brief Zero all memory variable (R) fields. */
  KOKKOS_FUNCTION
  void init() {
    this->Rxx = value_type(typename value_type::value_type(0));
    this->Ryy = value_type(typename value_type::value_type(0));
    this->Rxy = value_type(typename value_type::value_type(0));
    this->Rxz = value_type(typename value_type::value_type(0));
    this->Ryz = value_type(typename value_type::value_type(0));
    this->Rkappa = value_type(typename value_type::value_type(0));
  }

  /** @brief Component-wise addition on R fields. */
  KOKKOS_FUNCTION attenuation operator+(const attenuation &rhs) const {
    attenuation result;
    result.kappa_common_factor = this->kappa_common_factor;
    result.mu_common_factor = this->mu_common_factor;
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
    return result;
  }

  /** @brief In-place addition on R fields. */
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
    return *this;
  }

  /** @brief Scalar multiplication on R fields. */
  KOKKOS_FUNCTION attenuation operator*(const type_real &rhs) const {
    attenuation result;
    result.kappa_common_factor = this->kappa_common_factor;
    result.mu_common_factor = this->mu_common_factor;
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
    return result;
  }

  /** @brief Equality — compares all fields (factors and R). */
  KOKKOS_INLINE_FUNCTION
  bool operator==(const attenuation &other) const {
    return kappa_common_factor == other.kappa_common_factor &&
           mu_common_factor == other.mu_common_factor &&
           alpha_rk == other.alpha_rk && beta_rk == other.beta_rk &&
           gamma_rk == other.gamma_rk && Rxx == other.Rxx && Ryy == other.Ryy &&
           Rxy == other.Rxy && Rxz == other.Rxz && Ryz == other.Ryz &&
           Rkappa == other.Rkappa;
  }

  /** @brief String representation (non-SIMD only). */
  std::string print() const {
    std::ostringstream oss;
    oss << "Attenuation Factors:\n";
    for (int i = 0; i < N_SLS; ++i)
      oss << "  kappa_common_factor(" << i << ") = " << kappa_common_factor(i)
          << "\n";
    for (int i = 0; i < N_SLS; ++i)
      oss << "  mu_common_factor(" << i << ") = " << mu_common_factor(i)
          << "\n";
    oss << "  alpha_rk = " << alpha_rk << "\n";
    oss << "  beta_rk = " << beta_rk << "\n";
    oss << "  gamma_rk = " << gamma_rk << "\n";
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
    return oss.str();
  }
};

} // namespace specfem::point
