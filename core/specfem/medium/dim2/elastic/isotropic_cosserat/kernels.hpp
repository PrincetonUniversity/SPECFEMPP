#pragma once

#include "specfem/medium_container/impl/point_container.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::medium_container::kernels {

/**
 * @defgroup specfem_point_kernels_dim2_elastic_isotropic_cosserat 2D Cosserat
 * @{
 */

/**
 * @brief Placeholder container for 2D elastic isotropic cosserat media
 *
 * @tparam MediumTag The medium tag
 * @tparam UseSIMD Boolean indicating whether to use SIMD intrinsics
 */
template <specfem::element::medium_tag MediumTag, bool UseSIMD>
struct point_container<
    specfem::element::dimension_tag::dim2, MediumTag,
    specfem::element::property_tag::isotropic_cosserat, UseSIMD,
    std::enable_if_t<specfem::element::is_elastic<MediumTag>::value> >
    /// @cond
    : public KernelsAccessor<specfem::element::dimension_tag::dim2, MediumTag,
                             specfem::element::property_tag::isotropic_cosserat,
                             UseSIMD>
/// @endcond
{

  using base_type =
      KernelsAccessor<specfem::element::dimension_tag::dim2, MediumTag,
                      specfem::element::property_tag::isotropic_cosserat,
                      UseSIMD>;

  using value_type = typename base_type::value_type;
  using simd = typename base_type::simd;

  KOKKOS_INLINE_FUNCTION
  point_container() {
    Kokkos::abort(
        "Kernels container for elastic isotropic cosserat media is not "
        "implemented for this dimension");
  }
};

/** @} */ // end of group

} // namespace specfem::medium_container::kernels
