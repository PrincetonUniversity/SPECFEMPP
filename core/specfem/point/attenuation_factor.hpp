#include "specfem/constants.hpp"

namespace specfem::point {

template <typename Tags, bool UseSIMD> struct attenuation_factor;

/**
 * @brief Point structure to hold the attenuation factors for a specific GLL
 * point. Used to update the memory variables, R_mu and R_kappa, for the
 * constant isotropic attenuation case in elastic media.
 */
template <typename Tags, bool UseSIMD,
          std::enable_if_t<
              (Tags::dimension_tag == specfem::element::dimension_tag::dim3) &&
                  (Tags::medium_tag == specfem::element::medium_tag::elastic) &&
                  (Tags::attenuation_tag ==
                   specfem::element::attenuation_tag::constant_isotropic),
              int> = 0>
{

  constexpr static auto dimension_tag = Tags::dimension_tag;
  constexpr static auto medium_tag = Tags::medium_tag;
  constexpr static auto attenuation_tag = Tags::attenuation_tag;

private:
  using base_type = specfem::data_access::Accessor<
      specfem::datatype::AccessorType::point,
      specfem::data_access::DataClassType::attenuation_factor,
      Tags::dimension_tag,
      UseSIMD>; ///< Base type of the point attenuation factor

public:
  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = typename base_type::template simd<type_real>; ///< SIMD data type
  using value_type =
      typename base_type::template vector_type<type_real,
                                               specfem::constants::N_SLS>;
  ///@}

  value_type A_kappa; ///< @f$ A_\kappa^\ell =
                      ///< \dfrac{2\,\Delta\kappa_\ell}{\tau_\sigma^\ell} @f$
  value_type A_mu;    ///< @f$ A_\mu^\ell =
                      ///< \dfrac{2\,\Delta\mu_\ell}{\tau_\sigma^\ell} @f$
};

} // namespace specfem::point
