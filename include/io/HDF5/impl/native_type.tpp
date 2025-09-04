#pragma once

#ifndef NO_HDF5
#include "H5Cpp.h"
#endif

#include "native_type.hpp"
#include <iostream>

#ifndef NO_HDF5
template <> struct specfem::io::impl::HDF5::native_type<int> {
  static H5::IntType& type() {
    static H5::IntType type(H5::PredType::NATIVE_INT);
    type.setOrder(H5T_ORDER_LE);
    return type;
  }
};

template <> struct specfem::io::impl::HDF5::native_type<float> {
  static H5::FloatType& type() {
    static H5::FloatType type(H5::PredType::NATIVE_FLOAT);
    type.setOrder(H5T_ORDER_LE);
    return type;
  }
};

template <> struct specfem::io::impl::HDF5::native_type<double> {
  static H5::FloatType& type() {
    static H5::FloatType type(H5::PredType::NATIVE_DOUBLE);
    type.setOrder(H5T_ORDER_LE);
    return type;
  }
};

template <> struct specfem::io::impl::HDF5::native_type<long> {
  static H5::IntType& type() {
    static H5::IntType type(H5::PredType::NATIVE_LONG);
    type.setOrder(H5T_ORDER_LE);
    return type;
  }
};

template <> struct specfem::io::impl::HDF5::native_type<long long> {
  static H5::IntType& type() {
    static H5::IntType type(H5::PredType::NATIVE_LLONG);
    type.setOrder(H5T_ORDER_LE);
    return type;
  }
};

template <> struct specfem::io::impl::HDF5::native_type<unsigned int> {
  static H5::IntType& type() {
    static H5::IntType type(H5::PredType::NATIVE_UINT);
    type.setOrder(H5T_ORDER_LE);
    return type;
  }
};

template <> struct specfem::io::impl::HDF5::native_type<unsigned long> {
  static H5::IntType& type() {
    static H5::IntType type(H5::PredType::NATIVE_ULONG);
    type.setOrder(H5T_ORDER_LE);
    return type;
  }
};

template <> struct specfem::io::impl::HDF5::native_type<unsigned long long> {
  static H5::IntType& type() {
    static H5::IntType type(H5::PredType::NATIVE_ULLONG);
    type.setOrder(H5T_ORDER_LE);
    return type;
  }
};

template <> struct specfem::io::impl::HDF5::native_type<unsigned char> {
  static H5::IntType& type() {
    static H5::IntType type(H5::PredType::NATIVE_UCHAR);
    type.setOrder(H5T_ORDER_LE);
    return type;
  }
};

template <> struct specfem::io::impl::HDF5::native_type<char> {
  static H5::IntType& type() {
    static H5::IntType type(H5::PredType::NATIVE_CHAR);
    type.setOrder(H5T_ORDER_LE);
    return type;
  }
};

template <> struct specfem::io::impl::HDF5::native_type<short> {
  static H5::IntType& type() {
    static H5::IntType type(H5::PredType::NATIVE_SHORT);
    type.setOrder(H5T_ORDER_LE);
    return type;
  }
};

template <> struct specfem::io::impl::HDF5::native_type<unsigned short> {
  static H5::IntType& type() {
    static H5::IntType type(H5::PredType::NATIVE_USHORT);
    type.setOrder(H5T_ORDER_LE);
    return type;
  }
};

template <> struct specfem::io::impl::HDF5::native_type<bool> {
  static H5::IntType& type() {
    static H5::IntType type(H5::PredType::NATIVE_HBOOL);
    type.setOrder(H5T_ORDER_LE);
    return type;
  }
};

template <> struct specfem::io::impl::HDF5::native_type<std::string> {
  static H5::StrType& type() {
    static H5::StrType type(H5::PredType::C_S1, H5T_VARIABLE);
    type.setOrder(H5T_ORDER_LE);
    return type;
  }
};

#endif
