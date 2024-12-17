#pragma once
#include "acoustic_isotropic.hpp"
#include "elastic_anisotropic.hpp"
#include "elastic_isotropic.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "point/field.hpp"
#include "point/field_derivatives.hpp"
#include "point/kernels.hpp"
#include "point/properties.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace frechet_derivatives {
namespace impl {

template <specfem::dimension::type DimensionType,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool UseSIMD>
KOKKOS_FUNCTION
    specfem::point::kernels<DimensionType, MediumTag, PropertyTag, UseSIMD>
    element_kernel(
        const specfem::point::properties<DimensionType, MediumTag, PropertyTag,
                                         UseSIMD> &properties,
        const specfem::point::field<DimensionType, MediumTag, false, false,
                                    true, false, UseSIMD> &adjoint_field,
        const specfem::point::field<DimensionType, MediumTag, true, false,
                                    false, false, UseSIMD> &backward_field,
        const specfem::point::field_derivatives<DimensionType, MediumTag,
                                                UseSIMD> &adjoint_derivatives,
        const specfem::point::field_derivatives<DimensionType, MediumTag,
                                                UseSIMD> &backward_derivatives,
        const type_real &dt) {
  return impl_compute_element_kernel(properties, adjoint_field, backward_field,
                                     adjoint_derivatives, backward_derivatives,
                                     dt);
}

} // namespace impl
} // namespace frechet_derivatives
} // namespace specfem
