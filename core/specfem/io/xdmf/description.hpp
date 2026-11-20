#pragma once

#include "specfem/enums.hpp"
#include "specfem/mesh_entity.hpp"
#include <cstddef>
#include <string>
#include <vector>

namespace specfem {
namespace io {
namespace xdmf {

/**
 * @brief A per-GLL dataset stored in an element group of the heavy file.
 */
struct DatasetDescription {
  std::string name;                 ///< Dataset name within the group
  std::vector<std::size_t> extents; ///< Row-major extents of the dataset
};

/**
 * @brief A quantity computed by the XDMF reader from stored datasets.
 *
 * @p function is an XDMF `Function` expression in which `$i` refers to the
 * dataset named `operands[i]` of the same element group. Only `sqrt`,
 * `+ - * /`, parentheses and numeric literals are used. The Xdmf2 parser of
 * `vtkXdmfReader` only knows lower-case function names; it leaves an unknown
 * name such as `SQRT` unapplied without an error.
 */
struct DerivedDescription {
  std::string name;                  ///< Attribute name shown by the reader
  std::string function;              ///< XDMF Function expression
  std::vector<std::string> operands; ///< Dataset names bound to `$0`, `$1`, ...
};

/**
 * @brief Everything the XDMF sidecar needs to know about one element group.
 *
 * The group holds the coordinate datasets ("X", "Z" and, in 3D, "Y"), the
 * "connectivity" dataset written by
 * specfem::io::impl::sub_cell_connectivity, and the stored @p datasets.
 *
 * @tparam DimensionTag Dimension of the mesh
 */
template <specfem::element::dimension_tag DimensionTag>
struct ElementGroupDescription {
  std::string group; ///< Absolute group path, e.g. "/acoustic_isotropic"
  int number_of_elements = 0; ///< Number of elements in the group
  specfem::mesh_entity::element_grid<DimensionTag>
      element_grid = {};                    ///< GLL layout of every element
  std::vector<DatasetDescription> datasets; ///< Stored per-GLL datasets
  std::vector<DerivedDescription> derived;  ///< Quantities computed on read
};

/**
 * @brief Row-major extents of a per-GLL dataset of an element group.
 *
 * @tparam DimensionTag Dimension of the mesh
 * @param number_of_elements Number of elements in the group
 * @param element_grid GLL layout of every element
 * @return `{nelem, ngllz, ngllx}` in 2D, `{nelem, ngllz, nglly, ngllx}` in 3D
 */
template <specfem::element::dimension_tag DimensionTag>
std::vector<std::size_t> element_extents(
    const int number_of_elements,
    const specfem::mesh_entity::element_grid<DimensionTag> &element_grid) {
  if constexpr (DimensionTag == specfem::element::dimension_tag::dim2) {
    return { static_cast<std::size_t>(number_of_elements),
             static_cast<std::size_t>(element_grid.ngllz),
             static_cast<std::size_t>(element_grid.ngllx) };
  } else {
    return { static_cast<std::size_t>(number_of_elements),
             static_cast<std::size_t>(element_grid.ngllz),
             static_cast<std::size_t>(element_grid.nglly),
             static_cast<std::size_t>(element_grid.ngllx) };
  }
}

} // namespace xdmf
} // namespace io
} // namespace specfem
