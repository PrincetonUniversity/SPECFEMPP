#ifndef _SPECFEM_IO_ASCII_HPP
#define _SPECFEM_IO_ASCII_HPP

#include "impl/dataset.hpp"
#include "impl/dataset.tpp"
#include "impl/file.hpp"
#include "impl/group.hpp"

namespace specfem {
namespace io {

/**
 * @brief ASCII I/O backend for human-readable text format
 *
 * Provides File, Group, and Dataset abstractions using text files in folder
 * hierarchy. Each dataset creates .txt (data) and .meta (metadata) files for
 * easy inspection.
 *
 * @tparam OpType Operation type (specfem::io::read or specfem::io::write)
 */
template <typename OpType> class ASCII {
public:
  using IO_OpType = OpType; ///< Operation type (read/write)
  using File =
      specfem::io::impl::ASCII::File<OpType>; ///< ASCII file implementation
  using Group =
      specfem::io::impl::ASCII::Group<OpType>; ///< ASCII group implementation
  template <typename ViewType>
  using Dataset =
      specfem::io::impl::ASCII::Dataset<ViewType, OpType>; ///< ASCII dataset
                                                           ///< implementation
};

} // namespace io
} // namespace specfem

#endif /* _SPECFEM_IO_ASCII_HPP */
