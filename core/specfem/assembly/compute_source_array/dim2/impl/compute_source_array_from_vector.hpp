#pragma once

#include "specfem/assembly/mesh.hpp"
#include "specfem/source.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly::compute_source_array_impl {

/**
 * @brief Compute source array for a 2D vector source using Lagrange
 * interpolation.
 *
 * Algorithm:
 * 1. Extract GLL quadrature points from source_array dimensions
 * 2. Compute Lagrange interpolants at source location (xi, gamma)
 * 3. Distribute force vector to GLL points weighted by interpolant products
 *
 * Evaluates Lagrange basis functions at the source location and
 * distributes force vector components to GLL quadrature points.
 *
 * For a source at local coordinates @f$ (\xi_s, \gamma_s) @f$ with force
 * vector @f$ \mathbf{f} @f$, the source array is:
 * @f$ S_{i,jz,jx} = L_{jx}(\xi_s) L_{jz}(\gamma_s) f_i @f$
 *
 * where @f$ L_{jx}(\xi_s) @f$ is the Lagrange polynomial evaluated at the
 * source location.
 *
 * @param source Vector source containing force components and local coordinates
 * @param source_array Output array of shape (ncomponents, ngllz, ngllx)
 *
 * @note source_array values are non-zero only at the source's element location.
 * The magnitude at each GLL point equals the force vector component scaled by
 * the Lagrange interpolant product.
 */
void from_vector(
    const specfem::sources::vector_source<specfem::element::dimension_tag::dim2>
        &source,
    Kokkos::View<type_real ***, Kokkos::LayoutRight, Kokkos::HostSpace>
        source_array);

/**
 * @brief Accumulate a Lagrange-interpolated vector contribution onto a 2D
 * source array.
 *
 * For a vector @f$ \mathbf{v} @f$ located at reference coordinates
 * @f$ (\xi_s, \gamma_s) @f$, adds @f$ L_{jx}(\xi_s) L_{jz}(\gamma_s) v_i @f$
 * onto @c source_array in place (@c +=), where @f$ L @f$ are the Lagrange
 * interpolants at the source location. This is the shared kernel behind both
 * the force-source distribution (@ref from_vector) and the moment-tensor
 * monopole (body-couple) term; it multiplies the interpolant itself, not its
 * spatial gradient, so no mesh or Jacobian information is required.
 *
 * @param local_coordinates Reference coordinates (xi, gamma) of the source
 * @param vector Per-component vector to distribute; its extent must equal
 * @c source_array.extent(0) (a mismatch aborts)
 * @param source_array In/out array of shape (ncomponents, ngllz, ngllx);
 * contributions are accumulated (+=) onto existing values
 */
void accumulate_vector_contribution(
    const specfem::point::local_coordinates<
        specfem::element::dimension_tag::dim2> &local_coordinates,
    const Kokkos::View<type_real *, Kokkos::LayoutRight, Kokkos::HostSpace>
        &vector,
    Kokkos::View<type_real ***, Kokkos::LayoutRight, Kokkos::HostSpace>
        source_array);

} // namespace specfem::assembly::compute_source_array_impl
