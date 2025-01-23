#pragma once

#include "compute/assembly/assembly.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"

namespace specfem {
namespace kokkos_kernels {

namespace impl {
/**
 * @brief Compute the frechet derivatives for a particular material system.
 * This function is not defined as a private method of frechet_kernels due to
 * CUDA compatibility.
 *
 * @tparam MediumTag Medium tag.
 * @tparam PropertyTag Property tag.
 * @param dt Time interval.
 */
template <specfem::dimension::type DimensionType, int NGLL,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
void compute_material_derivatives(const specfem::compute::assembly &assembly,
                                  const type_real &dt);
} // namespace impl

} // namespace kokkos_kernels
} // namespace specfem
