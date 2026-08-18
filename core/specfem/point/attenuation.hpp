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
  using value_type = typename base_type::template scalar_type<type_real, N_SLS>;

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
  using value_type = typename base_type::template scalar_type<type_real, N_SLS>;

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
  // Type aliases
  // -----------------------------------------------------------------------
  using datatype = typename simd::datatype;

  // -----------------------------------------------------------------------
  // Data members — memory variables (dim2)
  // -----------------------------------------------------------------------
  value_type Rxx;
  value_type Rxz;
  value_type Rkappa;

  // -----------------------------------------------------------------------
  // Data members — symmetrised strain (Taylor step: ε(u + dt*v))
  //
  // Stores the 3 independent components of the symmetrised strain tensor
  // from the previous time step (used as Sn in the SLS update).
  // Replacing the raw 2×2 gradient tensor halves the per-point storage for
  // this field (4 → 3 scalars) which matters for GPU bandwidth.
  // -----------------------------------------------------------------------
  datatype epsilon_xx; ///< ε_xx = ∂u_x/∂x  (previous Taylor-step)
  datatype epsilon_zz; ///< ε_zz = ∂u_z/∂z  (previous Taylor-step)
  datatype epsilon_xz; ///< ε_xz = ½(∂u_x/∂z + ∂u_z/∂x)  (previous Taylor-step)

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
   * @param epsilon_xx          Stored symmetrised strain ε_xx
   * @param epsilon_zz          Stored symmetrised strain ε_zz
   * @param epsilon_xz          Stored symmetrised strain ε_xz
   */
  KOKKOS_FUNCTION
  attenuation(const value_type &kappa_relaxation_rate,
              const value_type &mu_relaxation_rate, const value_type &alpha_rk,
              const value_type &beta_rk, const value_type &gamma_rk,
              const value_type &Rxx, const value_type &Rxz,
              const value_type &Rkappa, const datatype &epsilon_xx,
              const datatype &epsilon_zz, const datatype &epsilon_xz)
      : kappa_relaxation_rate(kappa_relaxation_rate),
        mu_relaxation_rate(mu_relaxation_rate), alpha_rk(alpha_rk),
        beta_rk(beta_rk), gamma_rk(gamma_rk), Rxx(Rxx), Rxz(Rxz),
        Rkappa(Rkappa), epsilon_xx(epsilon_xx), epsilon_zz(epsilon_zz),
        epsilon_xz(epsilon_xz) {}

  // -----------------------------------------------------------------------
  // Methods
  // -----------------------------------------------------------------------

  /** @brief Zero all memory variable (R) fields and symmetrised strain. */
  KOKKOS_FUNCTION
  void init() {
    this->Rxx = value_type(typename value_type::value_type(0));
    this->Rxz = value_type(typename value_type::value_type(0));
    this->Rkappa = value_type(typename value_type::value_type(0));
    this->epsilon_xx = datatype(0);
    this->epsilon_zz = datatype(0);
    this->epsilon_xz = datatype(0);
  }

  /** @brief Component-wise addition on R fields and epsilon; RK/common factors
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
    result.epsilon_xx = this->epsilon_xx + rhs.epsilon_xx;
    result.epsilon_zz = this->epsilon_zz + rhs.epsilon_zz;
    result.epsilon_xz = this->epsilon_xz + rhs.epsilon_xz;
    return result;
  }

  /** @brief In-place addition on R fields and epsilon. */
  KOKKOS_FUNCTION attenuation &operator+=(const attenuation &rhs) {
    constexpr int N = specfem::constants::N_SLS;
    for (int i = 0; i < N; ++i) {
      this->Rxx(i) += rhs.Rxx(i);
      this->Rxz(i) += rhs.Rxz(i);
      this->Rkappa(i) += rhs.Rkappa(i);
    }
    this->epsilon_xx += rhs.epsilon_xx;
    this->epsilon_zz += rhs.epsilon_zz;
    this->epsilon_xz += rhs.epsilon_xz;
    return *this;
  }

  /** @brief Scalar multiplication on R fields and epsilon; RK/common factors
   * copied.
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
    result.epsilon_xx = this->epsilon_xx * rhs;
    result.epsilon_zz = this->epsilon_zz * rhs;
    result.epsilon_xz = this->epsilon_xz * rhs;
    return result;
  }

  /** @brief Equality — compares all fields (factors, R, and epsilon). */
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
    if (!specfem::utilities::is_close(epsilon_xx, other.epsilon_xx))
      return false;
    if (!specfem::utilities::is_close(epsilon_zz, other.epsilon_zz))
      return false;
    if (!specfem::utilities::is_close(epsilon_xz, other.epsilon_xz))
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
    oss << "Symmetrised strain (dim2):\n";
    oss << "  epsilon_xx = " << epsilon_xx << "\n";
    oss << "  epsilon_zz = " << epsilon_zz << "\n";
    oss << "  epsilon_xz = " << epsilon_xz << "\n";
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
  using value_type = typename base_type::template scalar_type<type_real, N_SLS>;

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
  // Type aliases
  // -----------------------------------------------------------------------
  using datatype = typename simd::datatype;

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
  // Data members — symmetrised strain (Taylor step: ε(u + dt*v))
  //
  // Stores the 6 independent components of the symmetrised strain tensor
  // from the previous time step (Sn for SLS update).
  // Replaces the raw 3×3 gradient tensor (9 → 6 scalars per GLL point).
  // -----------------------------------------------------------------------
  datatype epsilon_xx; ///< ε_xx = ∂u_x/∂x
  datatype epsilon_yy; ///< ε_yy = ∂u_y/∂y
  datatype epsilon_zz; ///< ε_zz = ∂u_z/∂z
  datatype epsilon_xy; ///< ε_xy = ½(∂u_x/∂y + ∂u_y/∂x)
  datatype epsilon_xz; ///< ε_xz = ½(∂u_x/∂z + ∂u_z/∂x)
  datatype epsilon_yz; ///< ε_yz = ½(∂u_y/∂z + ∂u_z/∂y)

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
   * @param epsilon_xx          Stored symmetrised strain ε_xx
   * @param epsilon_yy          Stored symmetrised strain ε_yy
   * @param epsilon_zz          Stored symmetrised strain ε_zz
   * @param epsilon_xy          Stored symmetrised strain ε_xy
   * @param epsilon_xz          Stored symmetrised strain ε_xz
   * @param epsilon_yz          Stored symmetrised strain ε_yz
   */
  KOKKOS_FUNCTION
  attenuation(const value_type &kappa_relaxation_rate,
              const value_type &mu_relaxation_rate, const value_type &alpha_rk,
              const value_type &beta_rk, const value_type &gamma_rk,
              const value_type &Rxx, const value_type &Ryy,
              const value_type &Rxy, const value_type &Rxz,
              const value_type &Ryz, const value_type &Rkappa,
              const datatype &epsilon_xx, const datatype &epsilon_yy,
              const datatype &epsilon_zz, const datatype &epsilon_xy,
              const datatype &epsilon_xz, const datatype &epsilon_yz)
      : kappa_relaxation_rate(kappa_relaxation_rate),
        mu_relaxation_rate(mu_relaxation_rate), alpha_rk(alpha_rk),
        beta_rk(beta_rk), gamma_rk(gamma_rk), Rxx(Rxx), Ryy(Ryy), Rxy(Rxy),
        Rxz(Rxz), Ryz(Ryz), Rkappa(Rkappa), epsilon_xx(epsilon_xx),
        epsilon_yy(epsilon_yy), epsilon_zz(epsilon_zz), epsilon_xy(epsilon_xy),
        epsilon_xz(epsilon_xz), epsilon_yz(epsilon_yz) {}

  // -----------------------------------------------------------------------
  // Methods
  // -----------------------------------------------------------------------

  /** @brief Zero all memory variable (R) fields and symmetrised strain. */
  KOKKOS_FUNCTION
  void init() {
    this->Rxx = value_type(typename value_type::value_type(0));
    this->Ryy = value_type(typename value_type::value_type(0));
    this->Rxy = value_type(typename value_type::value_type(0));
    this->Rxz = value_type(typename value_type::value_type(0));
    this->Ryz = value_type(typename value_type::value_type(0));
    this->Rkappa = value_type(typename value_type::value_type(0));
    this->epsilon_xx = datatype(0);
    this->epsilon_yy = datatype(0);
    this->epsilon_zz = datatype(0);
    this->epsilon_xy = datatype(0);
    this->epsilon_xz = datatype(0);
    this->epsilon_yz = datatype(0);
  }

  /** @brief Component-wise addition on R fields and epsilon; RK/common factors
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
    result.epsilon_xx = this->epsilon_xx + rhs.epsilon_xx;
    result.epsilon_yy = this->epsilon_yy + rhs.epsilon_yy;
    result.epsilon_zz = this->epsilon_zz + rhs.epsilon_zz;
    result.epsilon_xy = this->epsilon_xy + rhs.epsilon_xy;
    result.epsilon_xz = this->epsilon_xz + rhs.epsilon_xz;
    result.epsilon_yz = this->epsilon_yz + rhs.epsilon_yz;
    return result;
  }

  /** @brief In-place addition on R fields and epsilon. */
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
    this->epsilon_xx += rhs.epsilon_xx;
    this->epsilon_yy += rhs.epsilon_yy;
    this->epsilon_zz += rhs.epsilon_zz;
    this->epsilon_xy += rhs.epsilon_xy;
    this->epsilon_xz += rhs.epsilon_xz;
    this->epsilon_yz += rhs.epsilon_yz;
    return *this;
  }

  /** @brief Scalar multiplication on R fields and epsilon; RK/common factors
   * copied.
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
    result.epsilon_xx = this->epsilon_xx * rhs;
    result.epsilon_yy = this->epsilon_yy * rhs;
    result.epsilon_zz = this->epsilon_zz * rhs;
    result.epsilon_xy = this->epsilon_xy * rhs;
    result.epsilon_xz = this->epsilon_xz * rhs;
    result.epsilon_yz = this->epsilon_yz * rhs;
    return result;
  }

  /** @brief Equality — compares all fields (factors, R, and epsilon). */
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
    if (!specfem::utilities::is_close(epsilon_xx, other.epsilon_xx))
      return false;
    if (!specfem::utilities::is_close(epsilon_yy, other.epsilon_yy))
      return false;
    if (!specfem::utilities::is_close(epsilon_zz, other.epsilon_zz))
      return false;
    if (!specfem::utilities::is_close(epsilon_xy, other.epsilon_xy))
      return false;
    if (!specfem::utilities::is_close(epsilon_xz, other.epsilon_xz))
      return false;
    if (!specfem::utilities::is_close(epsilon_yz, other.epsilon_yz))
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
    oss << "Symmetrised strain (dim3):\n";
    oss << "  epsilon_xx = " << epsilon_xx << "\n";
    oss << "  epsilon_yy = " << epsilon_yy << "\n";
    oss << "  epsilon_zz = " << epsilon_zz << "\n";
    oss << "  epsilon_xy = " << epsilon_xy << "\n";
    oss << "  epsilon_xz = " << epsilon_xz << "\n";
    oss << "  epsilon_yz = " << epsilon_yz << "\n";
    return oss.str();
  }
};

} // namespace specfem::point
