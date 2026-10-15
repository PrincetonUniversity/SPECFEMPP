#pragma once

#include "specfem/assembly/boundaries.hpp"
#include "specfem/assembly/boundaries/dim3/impl/injection/injection_boundary.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/assembly/properties.hpp"
#include "specfem/injection/injection_inputs.hpp"
#include "specfem/injection/injection_provider.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>
#include <memory>

namespace specfem::assembly::boundaries_impl {

/**
 * @brief Build the Stacey-boundary injection field.
 *
 * Extracts the generic injection point set from the Stacey geometry and local
 * mesh material, constructs the injection provider for @c inputs.method, runs
 * its preprocessing (the method's own consistency check and field solve), and
 * fills @p out with the device injection state (point-index map, Stacey map,
 * and an initial frame-buffer snapshot). The owning provider is returned so the
 * caller (the assembly) can keep it host-side, apart from the device-captured
 * boundary container. A no-op (leaves @p out empty, returns @c nullptr) when
 * @c inputs.enabled is false or no Stacey points are present.
 *
 * @param inputs Injection domain inputs (method-specific payload).
 * @param stacey_data Stacey geometry container.
 * @param stacey_index_mapping Compute→local Stacey element map (device).
 * @param h_stacey_index_mapping Compute→local Stacey element map (host).
 * @param assembly_mesh Assembly mesh.
 * @param assembly_properties Material properties (local mesh moduli).
 * @param assembly_element_types Per-element medium/property classification.
 * @param dt Simulation time step in seconds.
 * @param nstep Total number of simulation time steps.
 * @param out Device injection state to populate.
 * @return The owning injection provider, or @c nullptr when disabled/empty.
 */
std::shared_ptr<specfem::injection::injection_provider<
    specfem::element::dimension_tag::dim3>>
make_injection_boundary(
    const specfem::injection::injection_inputs &inputs,
    const stacey<specfem::element::dimension_tag::dim3> &stacey_data,
    const Kokkos::View<int *, Kokkos::DefaultExecutionSpace>
        &stacey_index_mapping,
    const Kokkos::View<int *, Kokkos::DefaultExecutionSpace>::host_mirror_type
        &h_stacey_index_mapping,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3>
        &assembly_mesh,
    const specfem::assembly::properties<specfem::element::dimension_tag::dim3>
        &assembly_properties,
    const specfem::assembly::element_types<
        specfem::element::dimension_tag::dim3> &assembly_element_types,
    type_real dt, int nstep,
    injection_boundary<specfem::element::dimension_tag::dim3> &out);

} // namespace specfem::assembly::boundaries_impl
