#pragma once

#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

template <> struct specfem::io::impl::ASCII::native_type<bool> {
  constexpr static const char *string() { return "bool"; }
};

template <> struct specfem::io::impl::ASCII::native_type<unsigned short> {
  constexpr static const char *string() { return "unsigned short"; }
};

template <> struct specfem::io::impl::ASCII::native_type<short> {
  constexpr static const char *string() { return "short"; }
};

template <> struct specfem::io::impl::ASCII::native_type<int> {
  constexpr static const char *string() { return "int"; }
};

template <> struct specfem::io::impl::ASCII::native_type<long> {
  constexpr static const char *string() { return "long"; }
};

template <> struct specfem::io::impl::ASCII::native_type<long long> {
  constexpr static const char *string() { return "long long"; }
};

template <> struct specfem::io::impl::ASCII::native_type<unsigned int> {
  constexpr static const char *string() { return "unsigned int"; }
};

template <> struct specfem::io::impl::ASCII::native_type<unsigned long> {
  constexpr static const char *string() { return "unsigned long"; }
};

template <> struct specfem::io::impl::ASCII::native_type<unsigned long long> {
  constexpr static const char *string() { return "unsigned long long"; }
};

template <> struct specfem::io::impl::ASCII::native_type<unsigned char> {
  constexpr static const char *string() { return "unsigned char"; }
};

template <> struct specfem::io::impl::ASCII::native_type<float> {
  constexpr static const char *string() { return "float"; }
};

template <> struct specfem::io::impl::ASCII::native_type<double> {
  constexpr static const char *string() { return "double"; }
};
