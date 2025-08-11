#pragma once

#ifndef NO_ADIOS2
#include <adios2.h>
#endif

#include "io/operators.hpp"
#include "dataset.hpp"
#include "datasetbase.hpp"
#include "kokkos_abstractions.h"
#include "native_type.hpp"
#include "native_type.tpp"
#include <string>
#include <type_traits>
#include <vector>

#ifndef NO_ADIOS2

// Constructor implementation for write operations
template <typename ViewType, typename OpType>
template <typename T>
specfem::io::impl::ADIOS2::Dataset<ViewType, OpType>::Dataset(
    std::shared_ptr<adios2::IO> &io, std::shared_ptr<adios2::Engine> &engine,
    const std::string &name, const ViewType data,
    std::enable_if_t<std::is_same_v<T, specfem::io::write>, int>)
    : data(data), DatasetBase<OpType>(io, engine, name) {

  auto [shape, start, count] = this->convert_dimensions(data);

  // Define variable for writing
  using native_type = specfem::io::impl::ADIOS2::native_type<value_type>;
  variable = this->io_ptr->template DefineVariable<decltype(native_type::type())>(name, shape, start, count);
}

// Constructor implementation for read operations
template <typename ViewType, typename OpType>
template <typename T>
specfem::io::impl::ADIOS2::Dataset<ViewType, OpType>::Dataset(
    std::shared_ptr<adios2::IO> &io, std::shared_ptr<adios2::Engine> &engine,
    const std::string &name, const ViewType data,
    std::enable_if_t<std::is_same_v<T, specfem::io::read>, int>)
    : data(data), DatasetBase<OpType>(io, engine, name) {

  auto [shape, start, count] = this->convert_dimensions(data);

  // Inquire variable for reading
  using native_type = specfem::io::impl::ADIOS2::native_type<value_type>;
  variable = this->io_ptr->template InquireVariable<decltype(native_type::type())>(name);
  if (!variable) {
    throw std::runtime_error("Variable not found: " + name);
  }

  // Check dimensions match
  auto var_shape = variable.Shape();
  if (var_shape.size() != static_cast<size_t>(rank)) {
    throw std::runtime_error("Rank mismatch for variable: " + name);
  }

  for (int i = 0; i < rank; ++i) {
    if (var_shape[i] != shape[i]) {
      throw std::runtime_error("Dimension mismatch for variable: " + name);
    }
  }
}

// SFINAE write method implementation
template <typename ViewType, typename OpType>
template <typename T>
std::enable_if_t<std::is_same_v<T, specfem::io::write>, void>
specfem::io::impl::ADIOS2::Dataset<ViewType, OpType>::write() {

  using native_t = specfem::io::impl::ADIOS2::native_type<value_type>;
  using storage_type = decltype(native_t::type());

  if constexpr (std::is_same_v<value_type, storage_type>) {
    // Direct write - no conversion needed
    if (std::is_same_v<MemSpace, specfem::kokkos::HostMemSpace>) {
      DatasetBase<specfem::io::write>::write(data.data(), variable);
    } else if (std::is_same_v<MemSpace, specfem::kokkos::DevMemSpace>) {
      auto host_data = Kokkos::create_mirror_view(data);
      Kokkos::deep_copy(host_data, data);
      DatasetBase<specfem::io::write>::write(host_data.data(), variable);
    } else {
      throw std::runtime_error("Unknown memory space");
    }
  } else {
    // Convert then write
    native_t converter;
    if (std::is_same_v<MemSpace, specfem::kokkos::HostMemSpace>) {
      std::vector<storage_type> converted_data(data.size());
      for (size_t i = 0; i < data.size(); ++i) {
        converted_data[i] = converter(data.data()[i]);
      }
      DatasetBase<specfem::io::write>::write(converted_data.data(), variable);
    } else if (std::is_same_v<MemSpace, specfem::kokkos::DevMemSpace>) {
      auto host_data = Kokkos::create_mirror_view(data);
      Kokkos::deep_copy(host_data, data);
      std::vector<storage_type> converted_data(host_data.size());
      for (size_t i = 0; i < host_data.size(); ++i) {
        converted_data[i] = converter(host_data.data()[i]);
      }
      DatasetBase<specfem::io::write>::write(converted_data.data(), variable);
    } else {
      throw std::runtime_error("Unknown memory space");
    }
  }
}

// SFINAE read method implementation
template <typename ViewType, typename OpType>
template <typename T>
std::enable_if_t<std::is_same_v<T, specfem::io::read>, void>
specfem::io::impl::ADIOS2::Dataset<ViewType, OpType>::read() {

  using native_t = specfem::io::impl::ADIOS2::native_type<value_type>;
  using storage_type = decltype(native_t::type());

  if constexpr (std::is_same_v<value_type, storage_type>) {
    // Direct read - no conversion needed
    if (std::is_same_v<MemSpace, specfem::kokkos::HostMemSpace>) {
      DatasetBase<specfem::io::read>::read(data.data(), variable);
    } else if (std::is_same_v<MemSpace, specfem::kokkos::DevMemSpace>) {
      auto host_data = Kokkos::create_mirror_view(data);
      DatasetBase<specfem::io::read>::read(host_data.data(), variable);
      Kokkos::deep_copy(data, host_data);
    } else {
      throw std::runtime_error("Unknown memory space");
    }
  } else {
    // Read native type, then convert back
    if (std::is_same_v<MemSpace, specfem::kokkos::HostMemSpace>) {
      std::vector<storage_type> converted_data(data.size());
      DatasetBase<specfem::io::read>::read(converted_data.data(), variable);
      for (size_t i = 0; i < data.size(); ++i) {
        data.data()[i] = (converted_data[i] != 0);  // Convert back from storage type
      }
    } else if (std::is_same_v<MemSpace, specfem::kokkos::DevMemSpace>) {
      auto host_data = Kokkos::create_mirror_view(data);
      std::vector<storage_type> converted_data(host_data.size());
      DatasetBase<specfem::io::read>::read(converted_data.data(), variable);
      for (size_t i = 0; i < host_data.size(); ++i) {
        host_data.data()[i] = (converted_data[i] != 0);  // Convert back from storage type
      }
      Kokkos::deep_copy(data, host_data);
    } else {
      throw std::runtime_error("Unknown memory space");
    }
  }
}
#endif
