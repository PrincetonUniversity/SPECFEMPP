#pragma once

#include "specfem/medium_container/impl/point_container.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::point::impl::kernels {

/**
 * @defgroup specfem_point_kernels_dim2_electromagnetic 2D Electromagnetic
 * @{
 */

/**
 * @ingroup specfem_point_kernels_dim2_electromagnetic
 * @brief Data container to hold misfit kernels of 2D electromagnetic media at a
 * quadrature point
 *
 * @tparam MediumTag The medium tag
 * @tparam UseSIMD Boolean indicating whether to use SIMD intrinsics
 */
template <specfem::element::medium_tag MediumTag, bool UseSIMD>
struct point_container<
    specfem::dimension::type::dim2, MediumTag,
    specfem::element::property_tag::isotropic, UseSIMD,
    std::enable_if_t<specfem::element::is_electromagnetic<MediumTag>::value> >
    /// @cond
    : public KernelsAccessor<specfem::dimension::type::dim2, MediumTag,
                             specfem::element::property_tag::isotropic, UseSIMD>
/// @endcond
{
  using base_type =
      KernelsAccessor<specfem::dimension::type::dim2, MediumTag,
                      specfem::element::property_tag::isotropic, UseSIMD>;

  using value_type = typename base_type::value_type;
  using simd = typename base_type::simd;

  KOKKOS_INLINE_FUNCTION
  point_container() {
    Kokkos::abort("Kernels container for electromagnetic media is not "
                  "implemented for this dimension");
  }
};

/** @} */ // end of group

} // namespace specfem::point::impl::kernels
