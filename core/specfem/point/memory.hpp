

#include <Kokkos_Core.hpp>

namespace specfem::point {

/**
 * Point structure to hold the memory array for a specific GLL point. Used for
 * the attenuation computation in the memory kernel.
 *
 * Contains both
 */
template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::element::attenuation_tag AttenuationTag, bool UseSIMD>
    : struct Memory;

template <bool UseSIMD>
struct Memory<specfem::element::dim3, specfem::element::medium_tag::elastic,
              specfem::element::attenuation_tag::constant_isotropic, UseSIMD>
    : public specfem::data_access::Accessor<
          specfem::datatype::AccessorType::point,
          specfem::data_access::DataClassType::memory, DimensionTag, UseSIMD> {
private:
  /** @brief Base accessor type for data access framework integration */
  using base_type = specfem::data_access::Accessor<
      specfem::datatype::AccessorType::point,
      specfem::data_access::DataClassType::memory, DimensionTag, UseSIMD>;

public:
  /**
   * @name Static Properties
   * @brief Compile-time constants derived from template parameters
   */
  ///@{
  /** @brief Spatial dimension (2 or 3) */
  constexpr static int dimension =
      specfem::element::attributes<DimensionTag, MediumTag>::dimension;
  `

      /** @brief Number of components based on medium type */
      constexpr static int components =
          specfem::element::attributes<DimensionTag, MediumTag>::components;

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

  /** @brief Tensor type for storing stress components (components × dimension)
   */
  using value_type =
      typename base_type::template vector_type<type_real,
                                               specfem::constants::N_SLS>;
  ///@}

  /**
   * @name Data Members
   */
  ///@{
  /** @brief Memory variable tensor with dimensions (components × dimension ×
   * N_SLS) to store the memory variables R_mu and R_kappa for each standard
   * linear solid */
  value_type R_xx;
  value_type R_yy;
  value_type R_xy;
  value_type R_xz;
  value_type R_yz;
  value_type R_kappa;
  ///@}

  /**
   * @name Constructors
   */
  ///@{

  /**
   * @brief Default constructor.
   *
   * Initializes stress tensor with default values (typically zero).
   */
  KOKKOS_FUNCTION Memory() = default;

  /**
   * @brief Constructor with stress tensor initialization.
   *
   * @param T Stress tensor with components arranged as (components × dimension)
   *
   * @code{cpp}
   * // Example usage:
   * specfem::point::Memory<specfem::element::dim3,
   *                        specfem::element::medium_tag::elastic,
   *                        specfem::element::attenuation_tag::constant_isotropic,
   * true> point_memory_variable(R_mu_view, R_kappa_view);
   * @endcode
   */
  KOKKOS_FUNCTION Memory(const value_type &R_xx, const value_type &R_yy,
                         const value_type &R_xy, const value_type &R_xz,
                         const value_type &R_yz, const value_type &R_kappa)
      : R_xx(R_xx), R_yy(R_yy), R_xy(R_xy), R_xz(R_xz), R_yz(R_yz),
        R_kappa(R_kappa) {}
  ///@}

  // There should be a multiplication operator
}
