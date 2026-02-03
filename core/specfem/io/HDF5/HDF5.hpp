#pragma once

#ifndef NO_HDF5
#include "H5Cpp.h"
#endif

#include "specfem/io/HDF5/impl/dataset.hpp"
#include "specfem/io/HDF5/impl/dataset.tpp"
#include "specfem/io/HDF5/impl/file.hpp"
#include "specfem/io/HDF5/impl/group.hpp"

namespace specfem {
namespace io {
/**
 * @brief HDF5 I/O backend wrapper
 *
 * Template class providing File, Group, and Dataset abstractions for HDF5
 * format. Supports both read and write operations via OpType parameter.
 *
 * @tparam OpType Operation type (specfem::io::read or specfem::io::write)
 */
template <typename OpType> class HDF5 {
public:
  using IO_OpType = OpType; ///< Operation type (read/write)
  using File = specfem::io::impl::HDF5::File<OpType>; ///< Wrapper for HDF5 file
  using Group =
      specfem::io::impl::HDF5::Group<OpType>; ///< Wrapper for HDF5 group
  template <typename ViewType>
  using Dataset =
      specfem::io::impl::HDF5::Dataset<ViewType, OpType>; ///< Wrapper for HDF5
                                                          ///< dataset
};

} // namespace io
} // namespace specfem
