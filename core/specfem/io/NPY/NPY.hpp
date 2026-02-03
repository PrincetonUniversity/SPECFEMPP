#pragma once

#include "impl/dataset.hpp"
#include "impl/dataset.tpp"
#include "impl/file.hpp"
#include "impl/group.hpp"

namespace specfem::io {

/**
 * @brief NPY I/O backend for folder-based NumPy arrays
 *
 * Provides File, Group, and Dataset abstractions using folder structure with
 * .npy files. Compatible with NumPy's binary format for easy post-processing
 * with Python.
 *
 * @tparam OpType Operation type (specfem::io::read or specfem::io::write)
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
