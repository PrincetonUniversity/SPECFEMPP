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
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const specfem::assembly::element_types<specfem::dimension::type::dim2>
        &element_types,
    ContainerType &container);
} // namespace impl
} // namespace io
} // namespace specfem
