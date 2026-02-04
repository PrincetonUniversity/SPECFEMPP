#pragma once

#include "impl/dataset.hpp"
#include "impl/dataset.tpp"
#include "impl/file.hpp"
#include "impl/file.tpp"
#include "impl/group.hpp"

namespace specfem::io {

/**
 * @brief NPZ I/O backend for compressed NumPy archive format
 *
 * Provides File, Group, and Dataset abstractions for .npz files (ZIP archives
 * of .npy). Offers compact storage while maintaining NumPy compatibility.
 *
 * @tparam OpType Operation type (specfem::io::read or specfem::io::write)
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
