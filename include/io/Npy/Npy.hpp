#pragma once

#include "impl/dataset.hpp"
#include "impl/dataset.tpp"
#include "impl/file.hpp"
#include "impl/group.hpp"

namespace specfem::io {

/**
 * @brief
 *
 *
 * ASCII I/O writes the output in a human-readable format. The heirarchy of the
 * format is similar to that of HDF5 - where a file contains groups, which in
 * turn contain datasets. The implemetation creates separate directories for
 * each group and and separate .txt and .meta files for each dataset.
 *
 * @tparam OpType Operation type (read/write)
 */
template <typename OpType> class Npy {
public:
  using File = impl::Npy::File<OpType>;   ///< NPY file implementation
  using Group = impl::Npy::Group<OpType>; ///< NPY group implementation
  template <typename ViewType>
  using Dataset = impl::Npy::Dataset<ViewType, OpType>; ///< NPY dataset
                                                        ///< implementation
};

} // namespace specfem::io
