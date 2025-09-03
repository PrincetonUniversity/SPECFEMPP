#pragma once
#include "native_type.hpp"


#ifndef NO_ADIOS2
#include <adios2.h>
#include <cstdint>

// Generic template - ADIOS2 uses C++ types directly
template <> struct specfem::io::impl::ADIOS2::native_type<bool> {
  static std::uint8_t type() { return std::uint8_t{}; }
  std::uint8_t operator()(const bool& value) const { return value ? 1 : 0; }
};


#endif
