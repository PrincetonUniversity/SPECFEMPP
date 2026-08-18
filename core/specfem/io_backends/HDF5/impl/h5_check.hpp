#pragma once

#ifndef NO_HDF5

#include <hdf5.h>
#include <sstream>

#include "specfem/program.hpp"

namespace specfem {
namespace io_backends {
namespace impl {
namespace HDF5 {

/// @brief Check HDF5 return codes and abort on failure
inline void h5_check(herr_t err, const char *call, int line, const char *file) {
  if (err < 0) {
    std::ostringstream oss;
    oss << "HDF5 error in " << call << " at " << file << ":" << line;
    specfem::program::abort(oss.str(), 30, line, file);
  }
}

/// @brief Check HDF5 identifier return values and abort on failure
inline hid_t h5_check_id(hid_t id, const char *call, int line,
                         const char *file) {
  if (id < 0) {
    std::ostringstream oss;
    oss << "HDF5 error in " << call << " at " << file << ":" << line;
    specfem::program::abort(oss.str(), 30, line, file);
  }
  return id;
}

} // namespace HDF5
} // namespace impl
} // namespace io_backends
} // namespace specfem

#define SPECFEM_H5_CHECK(call)                                                 \
  specfem::io_backends::impl::HDF5::h5_check((call), #call, __LINE__, __FILE__)
#define SPECFEM_H5_CHECK_ID(call)                                              \
  specfem::io_backends::impl::HDF5::h5_check_id((call), #call, __LINE__,       \
                                                __FILE__)

#endif // NO_HDF5
