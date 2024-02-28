#include "IO/HDF5/impl/native_type.hpp"
#include "H5Cpp.h"
#include "IO/HDF5/impl/native_type.tpp"

// Explicit instantiation

template struct specfem::IO::impl::HDF5::native_type<float>;

template struct specfem::IO::impl::HDF5::native_type<double>;

template struct specfem::IO::impl::HDF5::native_type<int>;

template struct specfem::IO::impl::HDF5::native_type<long>;

template struct specfem::IO::impl::HDF5::native_type<long long>;

template struct specfem::IO::impl::HDF5::native_type<unsigned int>;

template struct specfem::IO::impl::HDF5::native_type<unsigned long>;

template struct specfem::IO::impl::HDF5::native_type<unsigned long long>;

template struct specfem::IO::impl::HDF5::native_type<short>;

template struct specfem::IO::impl::HDF5::native_type<unsigned short>;

template struct specfem::IO::impl::HDF5::native_type<char>;

template struct specfem::IO::impl::HDF5::native_type<unsigned char>;

template struct specfem::IO::impl::HDF5::native_type<bool>;
