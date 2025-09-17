#pragma once

#include "impl/dataset.hpp"
#include "impl/dataset.tpp"
#include "impl/file.hpp"
#include "impl/group.hpp"

namespace specfem::io {

/**
 * @brief Class template for handling folder-based NPY format operations
 *
 * NPY class provides an interface for reading from and writing to NPY format
 * files organized in a folder structure. NPY is a simple binary file format for
 * storing NumPy arrays. This implementation extends the basic format to support
 * a hierarchical structure similar to HDF5, where:
 * - Files represent the root container
 * - Groups represent directories/folders
 * - Datasets represent individual NPY files
 *
 * This allows for organized storage of multiple arrays in a structured
 * hierarchy.
 *
 * @tparam OpType Operation type (read/write) that determines the access mode
 */
template <typename OpType> class NPY {
public:
  using IO_OpType = OpType;               ///< Operation type (read/write)
  using File = impl::NPY::File<OpType>;   ///< NPY file implementation
  using Group = impl::NPY::Group<OpType>; ///< NPY group implementation
  template <typename ViewType>
  using Dataset = impl::NPY::Dataset<ViewType, OpType>; ///< NPY dataset
                                                        ///< implementation
};

} // namespace specfem::io
