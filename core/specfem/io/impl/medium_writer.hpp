#pragma once

namespace specfem {
namespace io {
namespace impl {

/**
 * @brief Write material property container to disk
 *
 * Generic function for outputting medium properties organized by element type.
 * Used internally by property_writer implementations.
 *
 * @tparam OutputLibrary I/O backend (HDF5, ASCII, etc.)
 * @tparam ContainerType Medium container type (elastic, acoustic, etc.)
 * @param output_folder Output location path
 * @param output_namespace Hierarchical namespace for organizing output
 * @param mesh Simulation mesh providing element information
 * @param element_types Element type organization structure
 * @param container Material property container to write
 */
template <typename OutputLibrary, typename ContainerType>
void write_container(
    const std::string &output_folder, const std::string &output_namespace,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim2> &mesh,
    const specfem::assembly::element_types<
        specfem::element::dimension_tag::dim2> &element_types,
    ContainerType &container);

template <typename GroupType, typename ElementIndicesType,
          typename DataContainerType>
int write_medium_group(
    GroupType &group,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim2> &mesh,
    const ElementIndicesType &element_indices,
    const DataContainerType &data_container);

/**
 * @brief Write the GLL coordinate datasets ("X"/"Z") for a group of elements.
 *
 * Shared between @ref write_medium_group and writers that assemble a group's
 * datasets themselves (e.g. the attenuation-aware property writer).
 *
 * @tparam GroupType Output group type of the I/O backend
 * @tparam ElementIndicesType Group-local index -> global ispec mapping type
 * @param group Output group to write coordinate datasets into
 * @param mesh Simulation mesh providing GLL coordinates
 * @param element_indices Group-local element index -> global ispec mapping
 */
template <typename GroupType, typename ElementIndicesType>
void write_coordinates(
    GroupType &group,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim2> &mesh,
    const ElementIndicesType &element_indices);

template <typename OutputLibrary, typename ContainerType>
void write_container(
    const std::string &output_folder, const std::string &output_namespace,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh,
    const specfem::assembly::element_types<
        specfem::element::dimension_tag::dim3> &element_types,
    ContainerType &container);

template <typename GroupType, typename ElementIndicesType,
          typename DataContainerType>
int write_medium_group(
    GroupType &group,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh,
    const ElementIndicesType &element_indices,
    const DataContainerType &data_container);

/**
 * @brief Write the GLL coordinate datasets ("X"/"Y"/"Z") for a group of
 *        elements.
 *
 * @tparam GroupType Output group type of the I/O backend
 * @tparam ElementIndicesType Group-local index -> global ispec mapping type
 * @param group Output group to write coordinate datasets into
 * @param mesh Simulation mesh providing GLL coordinates
 * @param element_indices Group-local element index -> global ispec mapping
 */
template <typename GroupType, typename ElementIndicesType>
void write_coordinates(
    GroupType &group,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh,
    const ElementIndicesType &element_indices);
} // namespace impl
} // namespace io
} // namespace specfem
