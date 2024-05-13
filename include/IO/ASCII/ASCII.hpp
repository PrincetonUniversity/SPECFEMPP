#ifndef _SPECFEM_IO_ASCII_HPP
#define _SPECFEM_IO_ASCII_HPP

#include "impl/dataset.hpp"
#include "impl/dataset.tpp"
#include "impl/file.hpp"
#include "impl/group.hpp"

namespace specfem {
namespace IO {

template <typename OpType> class ASCII {
public:
  using File = specfem::IO::impl::ASCII::File<OpType>;
  using Group = specfem::IO::impl::ASCII::Group<OpType>;
  template <typename ViewType>
  using Dataset = specfem::IO::impl::ASCII::Dataset<ViewType, OpType>;
};

} // namespace IO
} // namespace specfem

#endif /* _SPECFEM_IO_ASCII_HPP */
