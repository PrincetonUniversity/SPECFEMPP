#pragma once

#include "specfem/assembly.hpp"

namespace specfem::compute::impl {

/**
 * @brief Guest-side kernel of 3D nonconforming coupling.
 *
 * Works on the pairs of its own container, i.e. the list of
 * `(Tags::interface_tag, Tags::boundary_tag)`, whose self faces are the guest
 * and whose coupled faces are the host. For every pair it evaluates the
 * interface flux once at each host-face GLL node (the host's face factor and
 * outward normal, read from the own container, hence the sign flip) and adds
 * \f$ -\sum_c T_{c,k} \, \mathrm{flux}(c) \f$ to the acceleration at each
 * guest node \f$ k \f$, with \f$ T \f$ the guest-face interpolants stored
 * in the own container (see `specfem::algorithms::transfer_integrate`).
 * `Tags::boundary_tag` is the boundary tag of the guest (updated) element.
 * Returns without work when the own list is empty.
 *
 * @pre The container of `(Tags::interface_tag, Tags::boundary_tag)` is a guest
 * container (`self_is_host` is false). The dispatch in `compute_coupling.tpp`
 * guarantees this; the check is for direct callers.
 *
 * @throws std::runtime_error if the container is non-empty and its self side
 * is the host
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
