#pragma once

#include "specfem/element.hpp"
#include <string>

namespace specfem {
namespace io {
namespace impl {

/**
 * @brief No-op property I/O hook (default for write_container; kernel writer).
 *
 * Writes every view verbatim and appends no extra datasets.
 */
struct NoOpGroupWriter {
  template <specfem::element::medium_tag, specfem::element::property_tag,
            typename GroupType, typename ViewType>
  void write_view(GroupType &group, const std::string &name,
                  const ViewType &view) const {
    group.createDataset(name, view).write();
  }

  template <specfem::element::medium_tag, specfem::element::property_tag,
            typename GroupType>
  void attenuation_append(GroupType &) const {}
};

/**
 * @brief Write material property container to disk
 *
 * Generic function for outputting medium properties organized by element type.
 * Used internally by property_writer implementations.
 *
 * @tparam OutputLibrary I/O backend (HDF5, ASCII, etc.)
 * @tparam ContainerType Medium container type (elastic, acoustic, etc.)
 * @tparam ExtraGroupWriter Functor invoked per (medium, property) to append
 *         additional datasets to the group (defaults to a no-op).
 * @param output_folder Output location path
 * @param output_namespace Hierarchical namespace for organizing output
 * @param mesh Simulation mesh providing element information
 * @param element_types Element type organization structure
 * @param container Material property container to write
 * @param extra Extra-group writer functor
 */
template <typename OutputLibrary, typename ContainerType,
          typename ExtraGroupWriter = NoOpGroupWriter>
void write_container(
    const std::string &output_folder, const std::string &output_namespace,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim2> &mesh,
    const specfem::assembly::element_types<
        specfem::element::dimension_tag::dim2> &element_types,
    ContainerType &container, ExtraGroupWriter extra = {});

template <typename GroupType, typename ElementIndicesType,
          typename DataContainerType, typename ViewWriter>
int write_medium_group(
    GroupType &group,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim2> &mesh,
    const ElementIndicesType &element_indices,
    const DataContainerType &data_container, ViewWriter write_view);

template <typename OutputLibrary, typename ContainerType,
          typename ExtraGroupWriter = NoOpGroupWriter>
void write_container(
    const std::string &output_folder, const std::string &output_namespace,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh,
    const specfem::assembly::element_types<
        specfem::element::dimension_tag::dim3> &element_types,
    ContainerType &container, ExtraGroupWriter extra = {});

template <typename GroupType, typename ElementIndicesType,
          typename DataContainerType, typename ViewWriter>
int write_medium_group(
    GroupType &group,
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh,
    const ElementIndicesType &element_indices,
    const DataContainerType &data_container, ViewWriter write_view);
} // namespace impl
} // namespace io
} // namespace specfem
