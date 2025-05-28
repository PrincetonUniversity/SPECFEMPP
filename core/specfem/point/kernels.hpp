#pragma once
#include "impl/point_container.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace point {

namespace impl {
namespace kernels {

/**
 * @defgroup specfem_point_kernels_dim2_elastic_isotropic 2D Elastic Isotropic
 * Kernels
 * @{
 */
/**
 * @brief Data container to hold misfit kernels of 2D elastic isotropic media at
 * a quadrature point
 *
 * @tparam UseSIMD Boolean indicating whether to use SIMD intrinsics
 *
 * @fn const value_type rho() const
 *   @brief Get @f$ K_{\rho} @f$
 *   @return The value of @f$ K_{\rho} @f$
 *
 * @fn const value_type kappa() const
 *   @brief Get @f$ K_{\kappa} @f$
 *   @return The value of @f$ K_{\kappa} @f$
 *
 * @fn const value_type rhop() const
 *   @brief Get @f$ K_{\rho_p} @f$
 *   @return The value of @f$ K_{\rho_p} @f$
 *
 * @fn const value_type alpha() const
 *   @brief Get @f$ K_{\alpha} @f$
 *   @return The value of @f$ K_{\alpha} @f$
 *
 * @fn const value_type beta() const
 *   @brief Get @f$ K_{\beta} @f$
 *   @return The value of @f$ K_{\beta} @f$
 *
 */
template <specfem::element::medium_tag MediumTag, bool UseSIMD>
struct data_container<
    specfem::dimension::type::dim2, MediumTag,
    specfem::element::property_tag::isotropic, UseSIMD,
    std::enable_if_t<specfem::element::is_elastic<MediumTag>::value> >
    : public traits<specfem::dimension::type::dim2, MediumTag,
                    specfem::element::property_tag::isotropic, UseSIMD> {
  using base_type = traits<specfem::dimension::type::dim2, MediumTag,
                           specfem::element::property_tag::isotropic, UseSIMD>;

  using value_type = typename base_type::value_type;
  using simd = typename base_type::simd;

  POINT_CONTAINER(rho, mu, kappa, rhop, alpha, beta)
};
/** @} */ // end of group

/**
 * @brief Data container to hold misfit kernels of 2D elastic anisotropic media
 * at a quadrature point
 *
 * @tparam UseSIMD Boolean indicating whether to use SIMD intrinsics
 *
 * @fn const value_type rho() const
 *   @brief Get @f$ K_{\rho} @f$
 *   @return The value of @f$ K_{\rho} @f$
 *   @brief Get @f$ K_{c_{11}} @f$
 *
 * @fn const value_type c13() const
 *   @brief Get @f$ K_{c_{13}} @f$
 *   @return The value of @f$ K_{c_{13}} @f$
 *
 * @fn const value_type c15() const
 *   @brief Get @f$ K_{c_{15}} @f$
 *   @return The value of @f$ K_{c_{15}} @f$
 *
 * @fn const value_type c33() const
 *   @brief Get @f$ K_{c_{33}} @f$
 *   @return The value of @f$ K_{c_{33}} @f$
 *
 * @fn const value_type c35() const
 *   @brief Get @f$ K_{c_{35}} @f$
 *   @return The value of @f$ K_{c_{35}} @f$
 *
 * @fn const value_type c55() const
 *   @brief Get @f$ K_{c_{55}} @f$
 *   @return The value of @f$ K_{c_{55}} @f$
 */
template <specfem::element::medium_tag MediumTag, bool UseSIMD>
struct data_container<
    specfem::dimension::type::dim2, MediumTag,
    specfem::element::property_tag::anisotropic, UseSIMD,
    std::enable_if_t<specfem::element::is_elastic<MediumTag>::value> >
    : public traits<specfem::dimension::type::dim2, MediumTag,
                    specfem::element::property_tag::anisotropic, UseSIMD> {
  using base_type =
      traits<specfem::dimension::type::dim2, MediumTag,
             specfem::element::property_tag::anisotropic, UseSIMD>;

  using value_type = typename base_type::value_type;
  using simd = typename base_type::simd;

  POINT_CONTAINER(rho, c11, c13, c15, c33, c35, c55)
};

/**
 * @brief Data container to hold misfit kernels of 2D acoustic media at a
 * quadrature point
 *
 * @tparam UseSIMD Boolean indicating whether to use SIMD intrinsics
 *
 * @fn const value_type rho() const
 *   @brief Get @f$ K_{\rho} @f$
 *   @return The value of @f$ K_{\rho} @f$
 *
 * @fn const value_type kappa() const
 *   @brief Get @f$ K_{\kappa} @f$
 *   @return The value of @f$ K_{\kappa} @f$
 *
 * @fn const value_type rhop() const
 *   @brief Get @f$ K_{\rho_p} @f$
 *   @return The value of @f$ K_{\rho_p} @f$
 *
 * @fn const value_type alpha() const
 *   @brief Get @f$ K_{\alpha} @f$
 *   @return The value of @f$ K_{\alpha} @f$
 */
template <bool UseSIMD>
struct data_container<specfem::dimension::type::dim2,
                      specfem::element::medium_tag::acoustic,
                      specfem::element::property_tag::isotropic, UseSIMD>
    : public traits<specfem::dimension::type::dim2,
                    specfem::element::medium_tag::acoustic,
                    specfem::element::property_tag::isotropic, UseSIMD> {
  using base_type = traits<specfem::dimension::type::dim2,
                           specfem::element::medium_tag::acoustic,
                           specfem::element::property_tag::isotropic, UseSIMD>;

  using value_type = typename base_type::value_type;
  using simd = typename base_type::simd;

  POINT_CONTAINER(rho, kappa, rhop, alpha)

  KOKKOS_FUNCTION
  data_container(const value_type rho, const value_type kappa)
      : data_container(rho, kappa, rho * kappa,
                       static_cast<type_real>(2.0) * kappa) {}
};

template <specfem::element::medium_tag MediumTag, bool UseSIMD>
struct data_container<
    specfem::dimension::type::dim2, MediumTag,
    specfem::element::property_tag::isotropic, UseSIMD,
    std::enable_if_t<specfem::element::is_electromagnetic<MediumTag>::value> >
    : public traits<specfem::dimension::type::dim2, MediumTag,
                    specfem::element::property_tag::isotropic, UseSIMD> {
  using base_type = traits<specfem::dimension::type::dim2, MediumTag,
                           specfem::element::property_tag::isotropic, UseSIMD>;
  using value_type = typename base_type::value_type;
  using simd = typename base_type::simd;

  data_container() {
    Kokkos::abort("Kernels container for electromagnetic media is not "
                  "implemented for this dimension");
  }
};

/**
 * @brief Data container to hold misfit kernels of 2D poroelastic media at a
 * quadrature point
 *
 * @tparam UseSIMD Boolean indicating whether to use SIMD intrinsics
 *
 * @fn const value_type rhot() const
 *   @brief Get @f$ K_{\rho_t} @f$
 *   @return The value of @f$ K_{\rho_t} @f$
 *
 * @fn const value_type rhof() const
 *   @brief Get @f$ K_{\rho_f} @f$
 *   @return The value of @f$ K_{\rho_f} @f$
 *
 * @fn const value_type eta() const
 *   @brief Get @f$ K_{\eta} @f$
 *   @return The value of @f$ K_{\eta} @f$
 *
 * @fn const value_type sm() const
 *   @brief Get @f$ K_{s_m} @f$
 *   @return The value of @f$ K_{s_m} @f$
 *
 * @fn const value_type mu_fr() const
 *   @brief Get @f$ K_{\mu_{fr}} @f$
 *   @return The value of @f$ K_{\mu_{fr}} @f$
 *
 * @fn const value_type B() const
 *   @brief Get @f$ K_{B} @f$
 *   @return The value of @f$ K_{B} @f$
 *
 * @fn const value_type C() const
 *   @brief Get @f$ K_{C} @f$
 *   @return The value of @f$ K_{C} @f$
 *
 * @fn const value_type M() const
 *   @brief Get @f$ K_{M} @f$
 *   @return The value of @f$ K_{M} @f$
 *
 * @fn const value_type mu_frb() const
 *   @brief Get @f$ K_{\mu_{frb}} @f$
 *   @return The value of @f$ K_{\mu_{frb}} @f$
 *
 * @fn const value_type rhob() const
 *   @brief Get @f$ K_{\rho_b} @f$
 *   @return The value of @f$ K_{\rho_b} @f$
 *
 * @fn const value_type rhofb() const
 *   @brief Get @f$ K_{\rho_{fb}} @f$
 *   @return The value of @f$ K_{\rho_{fb}} @f$
 *
 * @fn const value_type phi() const
 *   @brief Get @f$ K_{\phi} @f$
 *   @return The value of @f$ K_{\phi} @f$
 *
 * @fn const value_type cpI() const
 *   @brief Get @f$ K_{cpI} @f$
 *   @return The value of @f$ K_{cpI} @f$
 *
 * @fn const value_type cpII() const
 *   @brief Get @f$ K_{cpII} @f$
 *   @return The value of @f$ K_{cpII} @f$
 *
 * @fn const value_type cs() const
 *   @brief Get @f$ K_{cs} @f$
 *   @return The value of @f$ K_{cs} @f$
 *
 * @fn const value_type rhobb() const
 *   @brief Get @f$ K_{\rho_{bb}} @f$
 *   @return The value of @f$ K_{\rho_{bb}} @f$
 *
 * @fn const value_type rhofbb() const
 *   @brief Get @f$ K_{\rho_{fbb}} @f$
 *   @return The value of @f$ K_{\rho_{fbb}} @f$
 *
 * @fn const value_type ratio() const
 *   @brief Get @f$ K_{ratio} @f$
 *   @return The value of @f$ K_{ratio} @f$
 *
 * @fn const value_type phib() const
 *   @brief Get @f$ K_{\phi_b} @f$
 *   @return The value of @f$ K_{\phi_b} @f$
 */
template <bool UseSIMD>
struct data_container<specfem::dimension::type::dim2,
                      specfem::element::medium_tag::poroelastic,
                      specfem::element::property_tag::isotropic, UseSIMD>
    : public traits<specfem::dimension::type::dim2,
                    specfem::element::medium_tag::poroelastic,
                    specfem::element::property_tag::isotropic, UseSIMD> {
  using base_type = traits<specfem::dimension::type::dim2,
                           specfem::element::medium_tag::poroelastic,
                           specfem::element::property_tag::isotropic, UseSIMD>;

  using value_type = typename base_type::value_type;
  using simd = typename base_type::simd;

  POINT_CONTAINER(rhot, rhof, eta, sm, mu_fr, B, C, M, mu_frb, rhob, rhofb, phi,
                  cpI, cpII, cs, rhobb, rhofbb, ratio, phib)

  KOKKOS_FUNCTION
  data_container(const value_type rhot, const value_type rhof,
                 const value_type eta, const value_type sm,
                 const value_type mu_fr, const value_type B, const value_type C,
                 const value_type M, const value_type cpI,
                 const value_type cpII, const value_type cs,
                 const value_type rhobb, const value_type rhofbb,
                 const value_type ratio, const value_type phib)
      : data_container(rhot, rhof, eta, sm, mu_fr, B, C, M, mu_fr,
                       (rhot + B + mu_fr), (rhof + C + M + sm),
                       (static_cast<value_type>(-1.0) * (sm + M)), cpI, cpII,
                       cs, rhobb, rhofbb, ratio, phib) {}
};

} // namespace kernels

} // namespace impl

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool UseSIMD,
          typename Enable = void>
struct kernels : public impl::kernels::data_container<DimensionType, MediumTag,
                                                      PropertyTag, UseSIMD> {
  using base_type = impl::kernels::data_container<DimensionType, MediumTag,
                                                  PropertyTag, UseSIMD>;

  using value_type = typename base_type::value_type;
  using simd = typename base_type::simd;

  using base_type::base_type;
};

} // namespace point
} // namespace specfem
