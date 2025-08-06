#pragma once

#ifndef NO_ADIOS2
#include <adios2.h>
#endif

#include "io/adios2/impl/dataset.hpp"
#include "io/adios2/impl/dataset.tpp"
#include "io/adios2/impl/file.hpp"
#include "io/adios2/impl/group.hpp"

namespace specfem {
namespace io {
/**
 * @brief ADIOS2 I/O wrapper
 *
 * @tparam OpType Operation type (read/write)
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
