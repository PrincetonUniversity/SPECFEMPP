#pragma once

#include "specfem/assembly/boundaries.hpp"
#include "specfem/assembly/boundaries/dim3/impl/injection/injection_boundary.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/assembly/properties.hpp"
#include "specfem/injection/injection_inputs.hpp"
#include "specfem/setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly::boundaries_impl {

/**
 * @brief Build the Stacey-boundary injection field.
 *
 * Extracts the generic injection point set from the Stacey geometry and local
 * mesh material, constructs the injection provider for @c inputs.method, runs
 * its preprocessing (the method's own consistency check and field solve), and
 * returns a populated @c injection_boundary with the provider and the device
 * point-index map. A no-op (returns an empty @c injection_boundary) when
 * @c inputs.enabled is false or no Stacey points are present.
 *
 * @param inputs Injection domain inputs (method-specific payload).
 * @param stacey_data Stacey geometry container.
 * @param h_stacey_index_mapping Compute→local Stacey element map (host).
 * @param assembly_mesh Assembly mesh.
 * @param assembly_properties Material properties (local mesh moduli).
 * @param assembly_element_types Per-element medium/property classification.
 * @param dt Simulation time step in seconds.
 * @param nstep Total number of simulation time steps.
 * @return A populated (or empty, if disabled) @c injection_boundary.
 */
injection_boundary<specfem::element::dimension_tag::dim3>
make_injection_boundary(
    const specfem::injection::injection_inputs &inputs,
    const stacey<specfem::element::dimension_tag::dim3> &stacey_data,
    const Kokkos::View<int *, Kokkos::DefaultExecutionSpace>::host_mirror_type
        &h_stacey_index_mapping,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3>
        &assembly_mesh,
    const specfem::assembly::properties<specfem::element::dimension_tag::dim3>
        &assembly_properties,
    const specfem::assembly::element_types<
        specfem::element::dimension_tag::dim3> &assembly_element_types,
    type_real dt, int nstep);

} // namespace specfem::assembly::boundaries_impl
