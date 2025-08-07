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
template <typename OpType> class ASCII {
public:
  using File = impl::ASCII::File<OpType>;   ///< ASCII file implementation
  using Group = impl::ASCII::Group<OpType>; ///< ASCII group implementation
  template <typename ViewType>
  using Dataset = impl::ASCII::Dataset<ViewType, OpType>; ///< ASCII dataset
                                                          ///< implementation
};

} // namespace specfem::io
