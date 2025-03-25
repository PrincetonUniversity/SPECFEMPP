#pragma once

#ifndef NO_HDF5
#include "H5Cpp.h"
#endif

#include "io/HDF5/impl/dataset.hpp"
#include "io/HDF5/impl/dataset.tpp"
#include "io/HDF5/impl/file.hpp"
#include "io/HDF5/impl/group.hpp"

namespace specfem {
namespace io {
/**
 * @brief HDF5 I/O wrapper
 *
 * @tparam OpType Operation type (read/write)
 */
template <typename OpType> class HDF5 {
public:
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
