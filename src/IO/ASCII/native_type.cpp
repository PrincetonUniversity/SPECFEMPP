#include "IO/ASCII/impl/native_type.hpp"
#include "IO/ASCII/impl/native_type.tpp"

// Explicit instantiation

template struct specfem::IO::impl::ASCII::native_type<float>;

template struct specfem::IO::impl::ASCII::native_type<double>;

template struct specfem::IO::impl::ASCII::native_type<int>;

template struct specfem::IO::impl::ASCII::native_type<long>;

template struct specfem::IO::impl::ASCII::native_type<long long>;

template struct specfem::IO::impl::ASCII::native_type<unsigned int>;

template struct specfem::IO::impl::ASCII::native_type<unsigned long>;

template struct specfem::IO::impl::ASCII::native_type<unsigned long long>;

template struct specfem::IO::impl::ASCII::native_type<short>;

template struct specfem::IO::impl::ASCII::native_type<unsigned short>;

// template struct specfem::IO::impl::ASCII::native_type<char>;

template struct specfem::IO::impl::ASCII::native_type<unsigned char>;

template struct specfem::IO::impl::ASCII::native_type<bool>;
