#pragma once

#include "specfem/assembly/assembly.hpp"
#include "specfem/io/reader.hpp"
#include "specfem/mesh.hpp"

#include <memory>

namespace specfem::assembly::dim3_impl {

/**
 * @brief Whether Cartesian mesh properties come from a property reader.
 *
 * @param property_reader Optional reader for a stored property model
 * @return true if @p property_reader is set
 */
bool has_deferred_properties(
    const specfem::mesh::cartesian3d_mesh &,
    const std::shared_ptr<specfem::io::reader> &property_reader);

/**
 * @brief Whether globe mesh properties are deferred.
 *
 * @return Always true: globe properties come from the model oracle or a
 * property reader.
 */
bool has_deferred_properties(const specfem::mesh::globe3d_mesh &,
                             const std::shared_ptr<specfem::io::reader> &);

/**
 * @brief Read deferred Cartesian mesh properties.
 *
 * @param assembly 3-D assembly whose property container is filled
 * @param property_reader Reader for the stored property model; nothing is read
 * if it is null
 */
void read_deferred_properties(
    const specfem::mesh::cartesian3d_mesh &,
    specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly,
    const std::shared_ptr<specfem::io::reader> &property_reader);

/**
 * @brief Read deferred globe mesh properties.
 *
 * Uses @p property_reader when it is set, and the globe model oracle
 * otherwise.
 *
 * @param mesh Raw Globe3D mesh
 * @param assembly 3-D assembly whose property container is filled
 * @param property_reader Optional reader for a stored property model
 */
void read_deferred_properties(
    const specfem::mesh::globe3d_mesh &mesh,
    specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly,
    const std::shared_ptr<specfem::io::reader> &property_reader);

/**
 * @brief Build element types for a Cartesian mesh.
 *
 * @param mesh Raw Cartesian mesh
 * @param assembly_mesh Assembled mesh
 * @return Element types without globe element context
 */
specfem::assembly::element_types<specfem::element::dimension_tag::dim3>
make_element_types(
    const specfem::mesh::cartesian3d_mesh &mesh,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3>
        &assembly_mesh);

/**
 * @brief Build element types for a globe mesh.
 *
 * @param mesh Raw Globe3D mesh
 * @param assembly_mesh Assembled mesh
 * @return Element types carrying the globe element context
 */
specfem::assembly::element_types<specfem::element::dimension_tag::dim3>
make_element_types(
    const specfem::mesh::globe3d_mesh &mesh,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3>
        &assembly_mesh);

/**
 * @brief Reference anchor coordinates of a Cartesian mesh.
 *
 * @return An empty view: Cartesian meshes have no reference geometry.
 */
specfem::mesh::control_nodes<
    specfem::element::dimension_tag::dim3>::CoordinatesViewType
reference_anchor_coordinates(const specfem::mesh::cartesian3d_mesh &);

/**
 * @brief Reference anchor coordinates of a globe mesh.
 *
 * @param mesh Raw Globe3D mesh
 * @return The globe reference coordinates, or an empty view if the mesh has
 * no reference geometry
 */
specfem::mesh::control_nodes<
    specfem::element::dimension_tag::dim3>::CoordinatesViewType
reference_anchor_coordinates(const specfem::mesh::globe3d_mesh &mesh);

} // namespace specfem::assembly::dim3_impl
