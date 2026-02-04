#pragma once

#ifndef NO_ADIOS2
#include <adios2.h>
#endif

#include "specfem/io/ADIOS2/impl/dataset.hpp"
#include "specfem/io/ADIOS2/impl/dataset.tpp"
#include "specfem/io/ADIOS2/impl/file.hpp"
#include "specfem/io/ADIOS2/impl/group.hpp"

namespace specfem {
namespace io {
/**
 * @brief ADIOS2 I/O backend wrapper
 *
 * Template class providing File, Group, and Dataset abstractions for ADIOS2
 * format. Designed for high-performance parallel I/O on supercomputers.
 *
 * @tparam OpType Operation type (specfem::io::read or specfem::io::write)
 */
template <typename OpType> class ADIOS2 {
public:
  using IO_OpType = OpType; ///< Operation type (read/write)
  using File =
      specfem::io::impl::ADIOS2::File<OpType>; ///< Wrapper for ADIOS2 file
  using Group =
      specfem::io::impl::ADIOS2::Group<OpType>; ///< Wrapper for ADIOS2 group
  template <typename ViewType>
  using Dataset =
      specfem::io::impl::ADIOS2::Dataset<ViewType, OpType>; ///< Wrapper for
                                                            ///< ADIOS2 dataset
};

} // namespace io
} // namespace specfem
