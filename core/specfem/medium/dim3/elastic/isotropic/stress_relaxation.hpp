#include "specfem/point/memory.hpp"
#include "specfem/point/stress.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::medium::physics::impl {
/**
 * @brief implements the addition of the attenuation term to the stress tensor
 * for elastic media
 *
 * Implement point stress + point memory variable operator
 */
template <typename Tags,
          std::enable_if_t<
              (Tags::dimension_tag == specfem::element::dimension_tag::dim3) &&
                  (Tags::medium_tag == specfem::element::medium_tag::elastic) &&
                  (Tags::attenuation_tag ==
                   specfem::element::attenuation_tag::constant_isotropic),
              int> = 0>
KOKKOS_INLINE_FUNCTION void compute_stress_relaxation(
    const specfem::point::memory<Tags::dimension_tag, Tags::medium_tag,
                                 Tags::attenuation_tag, UseSIMD>
        &point_memory_variable,
    specfem::point::stress<Tags::dimension_tag, Tags::medium_tag, UseSIMD>
        &point_stress, ) {

  // This should likely be an operator.
  for (int icomponent = 0; icomponent < components; ++icomponent) {
    for (int idimension = 0; idimension < dimension; ++idimension) {
      for (int isls = 0; isls < N_SLS; ++isls) {

        // Add the memory variable contribution to the stress tensor
        point_stress.T(icomponent, idimension) +=
            point_memory_variable.R_mu(isls, icomponent, idimension);

        // Add the volumetric contribution to the normal stress components
        if (icomponent == idimension) {
          point_stress.T(icomponent, idimension) +=
              point_memory_variable.R_kappa(isls, icomponent, idimension);
        }
      }
    }
  }

} // namespace specfem::medium::physics
