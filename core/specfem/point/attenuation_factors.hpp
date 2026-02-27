#include "specfem/constants.hpp"
#include "specfem/data_access/accessor.hpp"
#include "specfem/element.hpp"

namespace specfem::point {

template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::element::attenuation_tag AttenuationTag, bool UseSIMD>
struct attenuation_factors;

template <specfem::element::dimension_tag DimensionTag, bool UseSIMD>
struct attenuation_factors<
    DimensionTag, specfem::element::medium_tag::elastic,
    specfem::element::attenuation_tag::constant_isotropic, UseSIMD>
    : public specfem::data_access::Accessor<
          specfem::datatype::AccessorType::point,
          specfem::data_access::DataClassType::attenuation_factors,
          DimensionTag, UseSIMD> {
private:
  /** @brief Base accessor type for data access framework integration */
  using base_type = specfem::data_access::Accessor<
      specfem::datatype::AccessorType::point,
      specfem::data_access::DataClassType::attenuation_factors, DimensionTag,
      UseSIMD>;

public:
  /**
   * @name Static Properties
   * @brief Compile-time constants derived from template parameters
   */
  ///@{

  /** @brief Template parameter for spatial dimension */
  constexpr static specfem::element::dimension_tag dimension_tag = DimensionTag;

  /** @brief Template parameter for medium type */
  constexpr static specfem::element::attenuation_tag attenuation_tag =
      specfem::element::attenuation_tag::constant_isotropic;

  /** @brief Number of stress components based on medium type */
  constexpr static int N_SLS = specfem::constants::N_SLS;

  /** @brief Template parameter for SIMD usage */
  constexpr static bool using_simd = UseSIMD;
  ///@}

  /**
   * @name Type Definitions
   * @brief Type aliases for SIMD and tensor operations
   */
  ///@{
  /** @brief SIMD type for vectorized operations */
  using simd = typename base_type::template simd<type_real>;

  /**
   * @brief Vector type to store common factor for constant Q attenuation
   *        calculations (vector of length N_SLS)
   */
  using common_factor_type =
      typename base_type::template vector_type<type_real, N_SLS>;

  ///@}

  /**
   * @name Data Members
   */
  ///@{
  /** @brief kappa common factor for constant Q attenuation calculations */
  common_factor_type kappa_common_factor;
  common_factor_type mu_common_factor;

  /** @brief Runge Kutta integration factor for constant Q attenuation */
  type_real alpha_rk;
  type_real beta_rk;
  type_real gamma_rk;

  ///@}

  /**
   * @name Constructors
   */
  ///@{
  /**
   * @brief Default constructor.
   *
   * Initializes attenuation factors with default values (typically zero).
   */
  KOKKOS_FUNCTION attenuation_factors() = default;

  /**
   * @brief Constructor with attenuation factors initialization.
   *
   * @param kappa_common_factor Common factor for kappa values
   * @param mu_common_factor Common factor for mu values
   * @param alpha_rk Runge Kutta integration factor for alpha
   * @param beta_rk Runge Kutta integration factor for beta
   * @param gamma_rk Runge Kutta integration factor for gamma
   *
   * @code
   * // Example usage:
   * attenuation_factors<specfem::element::dimension_tag::dim2, false> factors(
   *     common_factor_type(1.0, 2.0),  // kappa_common_factor
   *     common_factor_type(3.0, 4.0),  // mu_common_factor
   *     0.5,                           // alpha_rk
   *     0.3,                           // beta_rk
   *     0.1                            // gamma_rk
   * );
   * @endcode
   */
  KOKKOS_FUNCTION
  attenuation_factors(const common_factor_type &kappa_common_factor,
                      const common_factor_type &mu_common_factor,
                      const type_real &alpha_rk, const type_real &beta_rk,
                      const type_real &gamma_rk)
      : kappa_common_factor(kappa_common_factor),
        mu_common_factor(mu_common_factor), alpha_rk(alpha_rk),
        beta_rk(beta_rk), gamma_rk(gamma_rk) {}
  ///@}

  /**
   * @brief Equality comparison operator.
   *
   */
  KOKKOS_INLINE_FUNCTION
  bool operator==(const attenuation_factors &other) const {
    return kappa_common_factor == other.kappa_common_factor &&
           mu_common_factor == other.mu_common_factor &&
           alpha_rk == other.alpha_rk && beta_rk == other.beta_rk &&
           gamma_rk == other.gamma_rk;
  };
  ///@}

  /**
   * @name Utility Functions
   */
  ///@{
  /**
   * @brief Generate string representation of the attenuation factors.
   *
   * Creates a formatted string showing all components of the attenuation
   * factors for debugging and visualization purposes. The output format shows
   * each component with its (component, dimension) indices.
   *
   * @return Formatted string representation of the attenuation factors
   *
   * @code
   * attenuation_factors<specfem::element::dimension_tag::dim2, false> factors(
   *     common_factor_type(1.0, 2.0),  // kappa_common_factor
   *     common_factor_type(3.0, 4.0),  // mu_common_factor
   *     0.5,                           // alpha_rk
   *     0.3,                           // beta_rk
   *     0.1                            // gamma_rk
   * );
   * std::cout << factors.print() << std::endl;
   * @endcode
   */
  std::string print() const {
    std::ostringstream oss;
    oss << "Attenuation Factors:\n";
    for (int i = 0; i < N_SLS; ++i) {
      oss << "  kappa_common_factor(" << i << ") = " << kappa_common_factor(i)
          << "\n";
    }
    for (int i = 0; i < N_SLS; ++i) {
      oss << "  mu_common_factor(" << i << ") = " << mu_common_factor(i)
          << "\n";
    }
    oss << "  alpha_rk = " << alpha_rk << "\n";
    oss << "  beta_rk = " << beta_rk << "\n";
    oss << "  gamma_rk = " << gamma_rk << "\n";
    return oss.str();
  }
  ///@}
};

} // namespace specfem::point
