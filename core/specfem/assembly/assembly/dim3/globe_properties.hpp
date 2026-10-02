#pragma once

#include "specfem/assembly/assembly.hpp"
#include "specfem/mesh.hpp"

namespace specfem::assembly::dim3_impl {

/**
 * @brief Populate GLL material properties by querying the globe evaluator.
 *
 * Globe raw meshes intentionally do not store pointwise material properties in
 * the thin database. This deferred setup uses the assembly's interpolated
 * reference GLL coordinates, passes those points and the element's
 * globe context from @c assembly.element_types to the SPECFEM3D_GLOBE model
 * evaluator, and writes the resulting density, wave speeds, and attenuation
 * values into the 3-D assembly properties container. Elastic isotropic values
 * use the oracle's Voigt average; acoustic values store inverse density.
 * Anisotropic elements receive all 21 Cartesian stiffness coefficients and
 * density, with no isotropic fallback.
 *
 * Evaluation is serial, in element batches (each batch evaluates all GLL
 * points). Properties are copied to device once after filling. Wall time,
 * element-call count and GLL-point count are logged per MPI rank.
 *
 * @param mesh Raw Globe3D mesh containing reference geometry and evaluator
 *        context
 * @param assembly 3-D assembly object whose property container is populated
 * @throws std::runtime_error if @c assembly.element_types carries no globe
 *         element context, or if the globe evaluator is unavailable or rejects
 *         a model/context combination
 */
void read_globe_properties(
    const specfem::mesh::globe3d_mesh &mesh,
    specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly);

/**
 * @brief Load a supplied GLL model, or populate properties from the oracle.
 * @param mesh Raw globe mesh containing model configuration.
 * @param assembly Assembly whose properties are populated.
 * @param property_reader Optional GLL model reader; takes precedence over the
 *        oracle, including its initialization and database validation.
 */
void read_deferred_properties(
    const specfem::mesh::globe3d_mesh &mesh,
    specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly,
    const std::shared_ptr<specfem::io::reader> &property_reader);

} // namespace specfem::assembly::dim3_impl
