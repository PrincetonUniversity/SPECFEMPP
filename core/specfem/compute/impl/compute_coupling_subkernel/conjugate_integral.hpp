#pragma once

#include "specfem/assembly.hpp"

namespace specfem::compute::impl {

/**
 * @brief Guest-side kernel of 3D nonconforming coupling.
 *
 * For every pair whose guest face is the self side of `Tags::interface_tag`,
 * evaluates the interface flux once at each host-face GLL node and adds
 * \f$ -\sum_c T_{c,k} \, \mathrm{flux}(c) \f$ to the acceleration at each
 * guest node \f$ k \f$, with \f$ T \f$ the guest-face interpolants stored by
 * the host container (see `specfem::algorithms::transfer_integrate`).
 * Returns without work when the container of the conjugate interface tag is
 * not the host.
 *
 * @tparam NGLL Number of GLL points per face axis
 * @tparam Tags Tags of the guest side (dimension, connection, wavefield,
 * interface, boundary, flux scheme)
 * @param assembly Assembly holding the fields, intersections and interface
 * data
 */
template <int NGLL, typename Tags>
void compute_coupling_conjugate_integral_nonconforming(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly)
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3);

} // namespace specfem::compute::impl
