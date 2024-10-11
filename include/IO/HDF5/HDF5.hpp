#pragma once

#ifndef NO_HDF5
#include "H5Cpp.h"
#endif

#include "IO/HDF5/impl/dataset.hpp"
#include "IO/HDF5/impl/dataset.tpp"
#include "IO/HDF5/impl/file.hpp"
#include "IO/HDF5/impl/group.hpp"

namespace specfem {
namespace IO {
/**
 * @brief HDF5 I/O wrapper
 *
 * @tparam OpType Operation type (read/write)
 */
template <typename OpType> class HDF5 {
public:
  using File = specfem::IO::impl::HDF5::File<OpType>; ///< Wrapper for HDF5 file
  using Group =
      specfem::IO::impl::HDF5::Group<OpType>; ///< Wrapper for HDF5 group
  template <typename ViewType>
  using Dataset =
      specfem::IO::impl::HDF5::Dataset<ViewType, OpType>; ///< Wrapper for HDF5
                                                          ///< dataset
};
} // namespace IO
} // namespace specfem
