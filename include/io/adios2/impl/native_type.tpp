#pragma once
#include "native_type.hpp"


#ifndef NO_ADIOS2
#include <adios2.h>
#include <cstdint>

// Generic template - ADIOS2 uses C++ types directly
template <> struct specfem::io::impl::ADIOS2::native_type<bool> {
  static std::uint8_t type() { return std::uint8_t{}; }  // ADIOS2 doesn't support bool, map to uint8_t
};

template <> struct specfem::io::impl::ADIOS2::native_type<int> {
  static int type() { return int{}; }  // ADIOS2 uses the C++ type directly
};

template <> struct specfem::io::impl::ADIOS2::native_type<float> {
  static float type() { return float{}; }  // ADIOS2 uses the C++ type directly
};

template <> struct specfem::io::impl::ADIOS2::native_type<double> {
  static double type() { return double{}; }  // ADIOS2 uses the C++ type directly
};

template <> struct specfem::io::impl::ADIOS2::native_type<long> {
  static long type() { return long{}; }  // ADIOS2 uses the C++ type directly
};

template <> struct specfem::io::impl::ADIOS2::native_type<long long> {
  static long long type() { return (long long){}; }  // ADIOS2 uses the C++ type directly
};

template <> struct specfem::io::impl::ADIOS2::native_type<unsigned int> {
  static unsigned int type() { return (unsigned int){}; }  // ADIOS2 uses the C++ type directly
};

template <> struct specfem::io::impl::ADIOS2::native_type<unsigned long> {
  static unsigned long type() { return (unsigned long){}; }  // ADIOS2 uses the C++ type directly
};

template <> struct specfem::io::impl::ADIOS2::native_type<unsigned long long> {
  static unsigned long long type() { return (unsigned long long){}; }  // ADIOS2 uses the C++ type directly
};

template <> struct specfem::io::impl::ADIOS2::native_type<std::string> {
  static std::string type() { return std::string{}; }  // ADIOS2 uses the C++ type directly
};

template <> struct specfem::io::impl::ADIOS2::native_type<char> {
  static char type() { return char{}; }  // ADIOS2 uses the C++ type directly
};

template <> struct specfem::io::impl::ADIOS2::native_type<unsigned char> {
  static unsigned char type() { return (unsigned char){}; }  // ADIOS2 uses the C++ type directly
};

template <> struct specfem::io::impl::ADIOS2::native_type<short> {
  static short type() { return short{}; }  // ADIOS2 uses the C++ type directly
};

template <> struct specfem::io::impl::ADIOS2::native_type<unsigned short> {
  static unsigned short type() { return (unsigned short){}; }  // ADIOS2 uses the C++ type directly
};

#endif
