#include "IO/HDF5/impl/native_type.hpp"
#include "H5Cpp.h"

template <>
const H5::PredType specfem::IO::impl::HDF5::native_type<float>::type =
    H5::PredType::NATIVE_FLOAT;

template <>
const H5::PredType specfem::IO::impl::HDF5::native_type<double>::type =
    H5::PredType::NATIVE_DOUBLE;

template <>
const H5::PredType specfem::IO::impl::HDF5::native_type<int>::type =
    H5::PredType::NATIVE_INT;

template <>
const H5::PredType specfem::IO::impl::HDF5::native_type<long>::type =
    H5::PredType::NATIVE_LONG;

template <>
const H5::PredType specfem::IO::impl::HDF5::native_type<long long>::type =
    H5::PredType::NATIVE_LLONG;

template <>
const H5::PredType specfem::IO::impl::HDF5::native_type<unsigned int>::type =
    H5::PredType::NATIVE_UINT;

template <>
const H5::PredType specfem::IO::impl::HDF5::native_type<unsigned long>::type =
    H5::PredType::NATIVE_ULONG;

template <>
const H5::PredType
    specfem::IO::impl::HDF5::native_type<unsigned long long>::type =
        H5::PredType::NATIVE_ULLONG;

template <>
const H5::PredType specfem::IO::impl::HDF5::native_type<short>::type =
    H5::PredType::NATIVE_SHORT;

template <>
const H5::PredType specfem::IO::impl::HDF5::native_type<unsigned short>::type =
    H5::PredType::NATIVE_USHORT;

template <>
const H5::PredType specfem::IO::impl::HDF5::native_type<char>::type =
    H5::PredType::NATIVE_CHAR;

template <>
const H5::PredType specfem::IO::impl::HDF5::native_type<unsigned char>::type =
    H5::PredType::NATIVE_UCHAR;

template <>
const H5::PredType specfem::IO::impl::HDF5::native_type<bool>::type =
    H5::PredType::NATIVE_HBOOL;
