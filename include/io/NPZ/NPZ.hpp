#pragma once

#include "impl/dataset.hpp"
#include "impl/dataset.tpp"
#include "impl/file.hpp"
#include "impl/file.tpp"
#include "impl/group.hpp"

namespace specfem::io {

/**
 * @brief
 *
 *
 * Zipped archive of NPY format.
 *
 * @tparam OpType Operation type (read/write)
 */
template <typename OpType> class NPZ {
public:
  using IO_OpType = OpType;               ///< Operation type (read/write)
  using File = impl::NPZ::File<OpType>;   ///< NPZ file implementation
  using Group = impl::NPZ::Group<OpType>; ///< NPZ group implementation
  template <typename ViewType>
  using Dataset = impl::NPZ::Dataset<ViewType, OpType>; ///< NPZ dataset
                                                        ///< implementation
};

} // namespace specfem::io
