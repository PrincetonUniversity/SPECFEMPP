#pragma once

#include "medium/properties_container.hpp"
#include "point/interface.hpp"
#include <Kokkos_SIMD.hpp>

namespace specfem {
namespace medium {

template <specfem::element::medium_tag MediumTag>
struct kernels_container<
    MediumTag, specfem::element::property_tag::isotropic_cosserat,
    std::enable_if_t<specfem::element::is_elastic<MediumTag>::value> >
    : public impl_kernels_container<
          MediumTag, specfem::element::property_tag::isotropic_cosserat, 1> {

  using base_type = impl_kernels_container<
      MediumTag, specfem::element::property_tag::isotropic_cosserat, 1>;

  using base_type::base_type;

  DEFINE_MEDIUM_VIEW(rho, 0) ///< density @f$ \rho @f$
};

} // namespace medium
} // namespace specfem
