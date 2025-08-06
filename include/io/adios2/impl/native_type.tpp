#pragma once
#include "native_type.hpp"


#ifndef NO_ADIOS2
#include <adios2.h>
#include <cstdint>

// Generic template - ADIOS2 uses C++ types directly
template <> struct specfem::io::impl::ADIOS2::native_type<bool> {
  using type = std::uint8_t;  // ADIOS2 doesn't support bool, map to uint8_t
};

template <> struct specfem::io::impl::ADIOS2::native_type<int> {
  using type = int;  // ADIOS2 uses the C++ type directly
};

template <> struct specfem::io::impl::ADIOS2::native_type<float> {
  using type = float;  // ADIOS2 uses the C++ type directly
};

template <> struct specfem::io::impl::ADIOS2::native_type<double> {
  using type = double;  // ADIOS2 uses the C++ type directly
};

template <> struct specfem::io::impl::ADIOS2::native_type<long> {
  using type = long;  // ADIOS2 uses the C++ type directly
};

template <> struct specfem::io::impl::ADIOS2::native_type<long long> {
  using type = long long;  // ADIOS2 uses the C++ type directly
};

template <> struct specfem::io::impl::ADIOS2::native_type<unsigned int> {
  using type = unsigned int;  // ADIOS2 uses the C++ type directly
};

template <> struct specfem::io::impl::ADIOS2::native_type<unsigned long> {
  using type = unsigned long;  // ADIOS2 uses the C++ type directly
};

template <> struct specfem::io::impl::ADIOS2::native_type<unsigned long long> {
  using type = unsigned long long;  // ADIOS2 uses the C++ type directly
};

template <> struct specfem::io::impl::ADIOS2::native_type<std::string> {
  using type = std::string;  // ADIOS2 uses the C++ type directly
};

template <> struct specfem::io::impl::ADIOS2::native_type<char> {
  using type = char;  // ADIOS2 uses the C++ type directly
};

template <> struct specfem::io::impl::ADIOS2::native_type<unsigned char> {
  using type = unsigned char;  // ADIOS2 uses the C++ type directly
};

template <> struct specfem::io::impl::ADIOS2::native_type<short> {
  using type = short;  // ADIOS2 uses its own String type
};

template <> struct specfem::io::impl::ADIOS2::native_type<unsigned short> {
  using type = unsigned short;  // ADIOS2 uses its own String type
};

#endif
