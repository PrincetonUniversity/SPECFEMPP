#pragma once

#include "medium/properties_container.hpp"
#include "point/interface.hpp"
#include <Kokkos_SIMD.hpp>

namespace specfem {
namespace medium {

template <specfem::element::medium_tag MediumTag>
struct properties_container<
    MediumTag, specfem::element::property_tag::isotropic_cosserat,
    std::enable_if_t<specfem::element::is_elastic<MediumTag>::value> >
    : public impl_properties_container<
          MediumTag, specfem::element::property_tag::isotropic_cosserat, 8> {
  using base_type = impl_properties_container<
      MediumTag, specfem::element::property_tag::isotropic_cosserat, 8>;
  using base_type::base_type;

  // Normal elastic properties
  DEFINE_MEDIUM_VIEW(rho, 0)   ///< density @f$ \rho @f$
  DEFINE_MEDIUM_VIEW(kappa, 1) ///< Bulk Modulus @f$ \lambda + 2\mu @f$
  DEFINE_MEDIUM_VIEW(mu, 2)    ///< shear modulus @f$ \mu @f$
  DEFINE_MEDIUM_VIEW(nu, 3)    ///< symmetry breaking modulus @f$ \nu @f$

  // Additional elastic properties for spin media _c for _couple
  DEFINE_MEDIUM_VIEW(j, 4)        ///< inertia density @f$ j @f$
  DEFINE_MEDIUM_VIEW(lambda_c, 5) ///< couple bulk modulus @f$ \kappa_c @f$
  DEFINE_MEDIUM_VIEW(mu_c, 6)     ///< couple shear modulus @f$ \mu_c @f$
  DEFINE_MEDIUM_VIEW(nu_c, 7)     ///< symmetry breaking modulus @f$ \nu_c @f$
};

} // namespace medium
} // namespace specfem
