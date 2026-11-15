#pragma once

#include "specfem/assembly/assembly.hpp"
#include "specfem/globe/model_evaluator.hpp"
#include "specfem/io/reader.hpp"
#include "specfem/mesh.hpp"

#include <memory>

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
 * For anisotropic elements, radial transverse-isotropic Love parameters are
 * converted to 21 Cartesian stiffness coefficients using the final GLL-point
 * orientation. Full-anisotropy coefficients from the evaluator are already in
 * Cartesian axes and are stored without another rotation. Attenuation is not
 * yet supported for anisotropic elements.
 *
 * Evaluation is serial, in element batches (each batch evaluates all GLL
 * points). Properties are copied to device once after filling. Wall time,
 * element-call count and GLL-point count are logged per MPI rank at debug
 * level.
 *
 * @param mesh Raw Globe3D mesh containing reference geometry and evaluator
 *        context
 * @param assembly 3-D assembly object whose property container is populated
 * @throws std::runtime_error if @c assembly.element_types carries no globe
 *         element context, if attenuation is enabled while any element is
 *         anisotropic, if the globe evaluator is unavailable or rejects a
 *         model/context combination, if the evaluator returns full anisotropy
 *         for an element without the anisotropic property tag, if a
 *         transverse-isotropic element is built without spherical
 *         coordinates, or if the evaluator returns non-zero azimuthal
 *         anisotropy (see @ref reject_azimuthal_anisotropy)
 */
void read_globe_properties(
    const specfem::mesh::globe3d_mesh &mesh,
    specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly);

/**
 * @brief Reject azimuthal anisotropy returned by the globe evaluator.
 *
 * The evaluator returns dimensionless \f$G_c'\f$ and \f$G_s'\f$ azimuthal
 * coefficients. For full anisotropy they are projections of the returned
 * Cartesian @c cij (globe's @c rotate_tensor_global_to_azi), which is stored
 * unchanged, so nothing is lost and they are ignored. For transverse isotropy
 * they would sit on top of the Love tensor; folding them into the stored 21
 * coefficients is deferred, so any non-zero value throws instead of being
 * silently dropped.
 *
 * @param values Evaluator output for one element
 * @param compute_ispec Compute element index, reported in the error message
 * @throws std::runtime_error if @c values is not fully anisotropic and any
 *         @c gc_prime or @c gs_prime is non-zero
 */
void reject_azimuthal_anisotropy(
    const specfem::globe::ModelEvaluator::ElementProperties &values,
    int compute_ispec);

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
