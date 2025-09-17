#pragma once

#ifndef NO_HDF5
#include "H5Cpp.h"
#else
#include <stdexcept>
#endif

namespace specfem {
namespace io {
namespace impl {
namespace HDF5 {

#ifdef NO_HDF5
template <typename T> struct native_type {
  static T type() {
    throw std::runtime_error("SPECFEM++ was not compiled with HDF5 support");
    return T();
  }
};
#else
template <typename T> struct native_type {};
#endif
} // namespace HDF5
} // namespace impl
} // namespace io
} // namespace specfem
