#pragma once

#include "specfem/assembly/assembly.hpp"
#include "specfem/mesh.hpp"

namespace specfem::assembly::dim3_impl {

/**
 * @brief Populate GLL material properties by querying the globe evaluator.
 *
 * Globe raw meshes intentionally do not store pointwise material properties in
 * the thin database. This deferred setup interpolates each element's stored
 * reference coordinates to GLL points, passes those points and the associated
 * @c globe_element_context to the SPECFEM3D_GLOBE model evaluator, and writes
 * the resulting density, wave speeds, and attenuation values into the 3-D
 * assembly properties container.
 *
 * @param mesh Raw Globe3D mesh containing reference geometry and evaluator
 *        context
 * @param assembly 3-D assembly object whose property container is populated
 * @throws std::runtime_error if the globe evaluator is unavailable or rejects a
 *         model/context combination
 */
void read_globe_properties(
    const specfem::mesh::globe3d_mesh &mesh,
    specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly);

} // namespace specfem::assembly::dim3_impl
