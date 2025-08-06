#ifndef SPECFEM_IO_ADIOS2_IMPL_NATIVE_TYPE_HPP
#define SPECFEM_IO_ADIOS2_IMPL_NATIVE_TYPE_HPP

#ifndef NO_ADIOS2
#include <adios2.h>
#else
#include <stdexcept>
#endif

namespace specfem {
namespace io {
namespace impl {
namespace ADIOS2 {

#ifdef NO_ADIOS2
template <typename T> struct native_type {
  static std::string type() {
    throw std::runtime_error("SPECFEM++ was not compiled with ADIOS2 support");
    return std::string();
  }
};
#else
template <typename T> struct native_type {};
#endif
} // namespace ADIOS2
} // namespace impl
} // namespace io
} // namespace specfem

#endif
