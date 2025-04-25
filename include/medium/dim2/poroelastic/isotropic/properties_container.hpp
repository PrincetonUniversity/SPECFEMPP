#pragma once

#include "medium/properties_container.hpp"
#include "point.hpp"
#include <Kokkos_SIMD.hpp>

namespace specfem {
namespace medium {

template <>
struct properties_container<specfem::element::medium_tag::poroelastic,
                            specfem::element::property_tag::isotropic>
    : public impl_properties_container<
          specfem::element::medium_tag::poroelastic,
          specfem::element::property_tag::isotropic, 12> {
  using base_type =
      impl_properties_container<specfem::element::medium_tag::poroelastic,
                                specfem::element::property_tag::isotropic, 12>;
  using base_type::base_type;

  DEFINE_MEDIUM_VIEW(phi, 0)        ///< porosity @f$ \phi @f$
  DEFINE_MEDIUM_VIEW(rho_s, 1)      ///< solid density @f$ \rho_s @f$
  DEFINE_MEDIUM_VIEW(rho_f, 2)      ///< fluid density @f$ \rho_f @f$
  DEFINE_MEDIUM_VIEW(tortuosity, 3) ///< tortuosity @f$ \tau @f$
  DEFINE_MEDIUM_VIEW(mu_G, 4)       ///  frame shear modulus
  DEFINE_MEDIUM_VIEW(H_Biot, 5)     /// Bulk moduli
  DEFINE_MEDIUM_VIEW(C_Biot, 6)     /// Solid/Fluid modulus
  DEFINE_MEDIUM_VIEW(M_Biot, 7)
  DEFINE_MEDIUM_VIEW(permxx, 8) /// permeability K
  DEFINE_MEDIUM_VIEW(permxz, 9)
  DEFINE_MEDIUM_VIEW(permzz, 10)
  DEFINE_MEDIUM_VIEW(eta_f, 11) ///< Viscosity @f$ \eta @f$
};

} // namespace medium
} // namespace specfem
