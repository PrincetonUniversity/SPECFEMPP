#pragma once
#include "impl/point_container.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace point {

namespace impl {
namespace kernels {

template <specfem::element::medium_tag MediumTag, bool UseSIMD>
struct data_container<
    specfem::dimension::type::dim2, MediumTag,
    specfem::element::property_tag::isotropic, UseSIMD,
    std::enable_if_t<specfem::element::is_elastic<MediumTag>::value> >
    : public point_traits<specfem::dimension::type::dim2, MediumTag,
                          specfem::element::property_tag::isotropic, UseSIMD> {
  using base_type =
      point_traits<specfem::dimension::type::dim2, MediumTag,
                   specfem::element::property_tag::isotropic, UseSIMD>;

  using value_type = typename base_type::value_type;
  using simd = typename base_type::simd;

  POINT_CONTAINER(rho, mu, kappa, rhop, alpha, beta)
};

template <specfem::element::medium_tag MediumTag, bool UseSIMD>
struct data_container<
    specfem::dimension::type::dim2, MediumTag,
    specfem::element::property_tag::anisotropic, UseSIMD,
    std::enable_if_t<specfem::element::is_elastic<MediumTag>::value> >
    : public point_traits<specfem::dimension::type::dim2, MediumTag,
                          specfem::element::property_tag::anisotropic,
                          UseSIMD> {
  using base_type =
      point_traits<specfem::dimension::type::dim2, MediumTag,
                   specfem::element::property_tag::anisotropic, UseSIMD>;

  using value_type = typename base_type::value_type;
  using simd = typename base_type::simd;

  POINT_CONTAINER(rho, c11, c13, c15, c33, c35, c55)
};

template <bool UseSIMD>
struct data_container<specfem::dimension::type::dim2,
                      specfem::element::medium_tag::acoustic,
                      specfem::element::property_tag::isotropic, UseSIMD>
    : public point_traits<specfem::dimension::type::dim2,
                          specfem::element::medium_tag::acoustic,
                          specfem::element::property_tag::isotropic, UseSIMD> {
  using base_type =
      point_traits<specfem::dimension::type::dim2,
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
    : public point_traits<specfem::dimension::type::dim2, MediumTag,
                          specfem::element::property_tag::isotropic, UseSIMD> {
  using base_type =
      point_traits<specfem::dimension::type::dim2, MediumTag,
                   specfem::element::property_tag::isotropic, UseSIMD>;
  using value_type = typename base_type::value_type;
  using simd = typename base_type::simd;

  data_container() {
    Kokkos::abort("Kernels container for electromagnetic media is not "
                  "implemented for this dimension");
  }
};

template <bool UseSIMD>
struct data_container<specfem::dimension::type::dim2,
                      specfem::element::medium_tag::poroelastic,
                      specfem::element::property_tag::isotropic, UseSIMD>
    : public point_traits<specfem::dimension::type::dim2,
                          specfem::element::medium_tag::poroelastic,
                          specfem::element::property_tag::isotropic, UseSIMD> {
  using base_type =
      point_traits<specfem::dimension::type::dim2,
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
