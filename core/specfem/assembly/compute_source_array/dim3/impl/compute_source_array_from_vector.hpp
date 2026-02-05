#pragma once
#include "kokkos_abstractions.h"
#include "specfem/assembly/mesh.hpp"
#include "specfem/source.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly::compute_source_array_impl {

/**
 * @brief Compute source array for a 3D vector source using Lagrange
 * interpolation.
 *
 * Algorithm:
 * 1. Extract GLL quadrature points from source_array dimensions
 * 2. Compute Lagrange interpolants at source location (xi, eta, gamma)
 * 3. Distribute force vector to GLL points weighted by Lagrange interpolants
 *
 * Evaluates Lagrange basis functions at the source location and
 * distributes force vector components to GLL quadrature points.
 *
 * For a source at local coordinates @f$ (\xi_s, \eta_s, \gamma_s) @f$ with
 * force vector @f$ \mathbf{f} @f$, the source array is:
 * @f$ S_{i,jz,jy,jx} = L_{jx}(\xi_s) L_{jy}(\eta_s) L_{jz}(\gamma_s) f_i @f$
 *
 * where @f$ L_{jx}(\xi_s) @f$ is the Lagrange polynomial evaluated at the
 * source location.
 *
 * @param source Vector source containing force components and local coordinates
 * @param source_array Output array of shape (ncomponents, ngllz, nglly, ngllx)
 *
 * @note source_array values are non-zero only at the source's element location.
 * The magnitude at each GLL point equals the force vector component scaled by
 * the Lagrange interpolant product.
 */
void from_vector(
    const specfem::sources::vector_source<specfem::element::dimension_tag::dim3>
        &source,
    Kokkos::View<type_real ****, Kokkos::LayoutRight, Kokkos::HostSpace>
        source_array);

} // namespace specfem::assembly::compute_source_array_impl
