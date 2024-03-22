#ifndef SPECFEM_IO_HDF5_HPP
#define SPECFEM_IO_HDF5_HPP

#include "H5Cpp.h"
#include "IO/HDF5/impl/dataset.hpp"
#include "IO/HDF5/impl/dataset.tpp"
#include "IO/HDF5/impl/file.hpp"
#include "IO/HDF5/impl/group.hpp"

namespace specfem {
namespace IO {
template <typename OpType> class HDF5 {
public:
  using File = specfem::IO::impl::HDF5::File<OpType>;
  using Group = specfem::IO::impl::HDF5::Group<OpType>;
  template <typename ViewType>
  using Dataset = specfem::IO::impl::HDF5::Dataset<ViewType, OpType>;
};
} // namespace IO
} // namespace specfem

#endif /* SPECFEM_IO_HDF5_HPP */
