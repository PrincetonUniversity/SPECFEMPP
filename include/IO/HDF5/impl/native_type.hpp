#ifndef SPECFEM_IO_HDF5_IMPL_NATIVE_TYPE_HPP
#define SPECFEM_IO_HDF5_IMPL_NATIVE_TYPE_HPP

#include "H5Cpp.h"

namespace specfem {
namespace IO {
namespace impl {
namespace HDF5 {

template <typename T> struct native_type {};
} // namespace HDF5
} // namespace impl
} // namespace IO
} // namespace specfem

#endif
