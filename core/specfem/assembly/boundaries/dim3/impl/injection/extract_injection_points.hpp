#pragma once

#include "specfem/assembly/boundaries.hpp"
#include "specfem/assembly/boundaries/dim3/impl/injection/injection_boundary.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/assembly/properties.hpp"
#include "specfem/injection/injection_points.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly::boundaries_impl {

/**
 * @brief Enumerate the Stacey boundary GLL points into a generic point set.
 *
 * Walks every local Stacey element and every GLL point tagged
 * @c boundary_tag::stacey (membership, so composite Stacey points are
 * included). For each such point it records the global coordinates (with
 * @c z converted to depth, positive-down, as @c mesh.zmax - z), the normalized
 * outward normal (the stored face normal is @c J*grad(xi), not unit), the
 * element medium, and the local isotropic material @c (rho, vp, vs) read from
 * @p assembly_properties. It also fills @p h_point_index (initialized to
 * @c -1) with each point's row in the returned set, keyed identically to
 * @c stacey::quadrature_point_boundary_tag.
 *
 * @param stacey_data Stacey geometry container (tags, normals, local indexing).
 * @param h_stacey_index_mapping Compute→local Stacey element map (host).
 * @param assembly_mesh Assembly mesh (coordinates, bounds, GLL extents).
 * @param assembly_properties Material properties (local mesh moduli).
 * @param assembly_element_types Per-element medium/property classification.
 * @param h_point_index Host point-index map to fill (sized like the Stacey tag
 *        view); set to @c -1 everywhere first, then to the point row.
 * @return The method-agnostic injection point set in row order.
 * @throws std::runtime_error on an unsupported medium/property combination.
 */
specfem::injection::injection_points extract_stacey_injection_points(
    const stacey<specfem::element::dimension_tag::dim3> &stacey_data,
    const Kokkos::View<int *, Kokkos::DefaultExecutionSpace>::host_mirror_type
        &h_stacey_index_mapping,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3>
        &assembly_mesh,
    const specfem::assembly::properties<specfem::element::dimension_tag::dim3>
        &assembly_properties,
    const specfem::assembly::element_types<
        specfem::element::dimension_tag::dim3> &assembly_element_types,
    const injection_boundary<specfem::element::dimension_tag::dim3>::
        PointIndexView::host_mirror_type &h_point_index);

} // namespace specfem::assembly::boundaries_impl
