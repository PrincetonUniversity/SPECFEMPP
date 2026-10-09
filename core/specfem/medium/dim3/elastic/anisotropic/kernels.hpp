#pragma once

#include "specfem/medium_container/impl/point_container.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::medium_container::kernels {

/**
 * @brief Point misfit kernels for a 3D fully anisotropic elastic medium.
 *
 * The stiffness components follow the Voigt order
 * $(11,12,13,14,15,16,22,23,24,25,26,33,34,35,36,44,45,46,55,56,66)$.
 *
 * @tparam MediumTag Physical medium type; must be elastic.
 * @tparam UseSIMD Whether values use SIMD lanes.
 */
template <specfem::element::medium_tag MediumTag, bool UseSIMD>
struct point_container<
    specfem::element::dimension_tag::dim3, MediumTag,
    specfem::element::property_tag::anisotropic, UseSIMD,
    std::enable_if_t<specfem::element::is_elastic<MediumTag>::value>>
    : public KernelsAccessor<specfem::element::dimension_tag::dim3, MediumTag,
                             specfem::element::property_tag::anisotropic,
                             UseSIMD> {
private:
  using base_type =
      KernelsAccessor<specfem::element::dimension_tag::dim3, MediumTag,
                      specfem::element::property_tag::anisotropic, UseSIMD>;

public:
  using value_type = typename base_type::value_type;
  using simd = typename base_type::simd;

  POINT_CONTAINER(rho, c11, c12, c13, c14, c15, c16, c22, c23, c24, c25, c26,
                  c33, c34, c35, c36, c44, c45, c46, c55, c56, c66)
};

} // namespace specfem::medium_container::kernels
