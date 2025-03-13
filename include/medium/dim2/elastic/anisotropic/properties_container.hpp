#pragma once

#include "medium/properties_container.hpp"
#include "point/interface.hpp"
#include <Kokkos_SIMD.hpp>

namespace specfem {
namespace medium {

template <specfem::element::medium_tag MediumTag>
struct properties_container<MediumTag,
                            specfem::element::property_tag::anisotropic>
    : public impl_properties_container<
          MediumTag, specfem::element::property_tag::anisotropic, 10,
          std::enable_if_t<
              MediumTag == specfem::element::medium_tag::elastic_sh ||
              MediumTag == specfem::element::medium_tag::elastic_sv> > {
  using base_type = impl_properties_container<
      MediumTag, specfem::element::property_tag::anisotropic, 10>;
  using base_type::base_type;

  DEFINE_MEDIUM_VIEW(c11, 0)
  DEFINE_MEDIUM_VIEW(c13, 1)
  DEFINE_MEDIUM_VIEW(c15, 2)
  DEFINE_MEDIUM_VIEW(c33, 3)
  DEFINE_MEDIUM_VIEW(c35, 4)
  DEFINE_MEDIUM_VIEW(c55, 5)
  DEFINE_MEDIUM_VIEW(c12, 6)
  DEFINE_MEDIUM_VIEW(c23, 7)
  DEFINE_MEDIUM_VIEW(c25, 8)
  DEFINE_MEDIUM_VIEW(rho, 9)
};

} // namespace medium
} // namespace specfem
