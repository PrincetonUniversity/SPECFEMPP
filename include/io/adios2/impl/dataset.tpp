#ifndef SPECFEM_IO_ADIOS2_IMPL_DATASET_TPP
#define SPECFEM_IO_ADIOS2_IMPL_DATASET_TPP

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
template <typename ViewType, typename OpType>
specfem::io::impl::ADIOS2::Dataset<ViewType, OpType>::Dataset(
    std::shared_ptr<adios2::IO> &io, std::shared_ptr<adios2::Engine> &engine,
    const std::string &name, const ViewType data)
    : data(data), DatasetBase<OpType>(io, engine, name) {

  // Convert dimensions to ADIOS2 format
  std::vector<std::size_t> shape(rank);
  std::vector<std::size_t> start(rank, 0);
  std::vector<std::size_t> count(rank);

  for (int i = 0; i < rank; ++i) {
    shape[i] = data.extent(i);
    count[i] = data.extent(i);  // full dataset write
  }

  if constexpr (std::is_same_v<OpType, specfem::io::write>) {
    // Define variable for writing
    variable = this->io_ptr->template DefineVariable<typename native_type::type>(name, shape, start, count);
  } else {
    // Inquire variable for reading
    variable = this->io_ptr->template InquireVariable<typename native_type::type>(name);
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
}

template <typename ViewType, typename OpType>
void specfem::io::impl::ADIOS2::Dataset<ViewType, OpType>::write() {
  static_assert(std::is_same_v<OpType, specfem::io::write>,
                "write() can only be called on write datasets");

  using storage_type = typename native_type::type;

  if constexpr (std::is_same_v<value_type, bool> && !std::is_same_v<storage_type, bool>) {
    // Handle bool conversion to uint8_t
    if (std::is_same_v<MemSpace, specfem::kokkos::HostMemSpace>) {
      std::vector<storage_type> converted_data(data.size());
      for (size_t i = 0; i < data.size(); ++i) {
        converted_data[i] = static_cast<storage_type>(data.data()[i] ? 1 : 0);
      }
      DatasetBase<OpType>::write(converted_data.data(), variable);
    } else if (std::is_same_v<MemSpace, specfem::kokkos::DevMemSpace>) {
      auto host_data = Kokkos::create_mirror_view(data);
      Kokkos::deep_copy(host_data, data);
      std::vector<storage_type> converted_data(host_data.size());
      for (size_t i = 0; i < host_data.size(); ++i) {
        converted_data[i] = static_cast<storage_type>(host_data.data()[i] ? 1 : 0);
      }
      DatasetBase<OpType>::write(converted_data.data(), variable);
    } else {
      throw std::runtime_error("Unknown memory space");
    }
  } else {
    // Standard case - no conversion needed
    if (std::is_same_v<MemSpace, specfem::kokkos::HostMemSpace>) {
      DatasetBase<OpType>::write(data.data(), variable);
    } else if (std::is_same_v<MemSpace, specfem::kokkos::DevMemSpace>) {
      auto host_data = Kokkos::create_mirror_view(data);
      Kokkos::deep_copy(host_data, data);
      DatasetBase<OpType>::write(host_data.data(), variable);
    } else {
      throw std::runtime_error("Unknown memory space");
    }
  }
}

template <typename ViewType, typename OpType>
void specfem::io::impl::ADIOS2::Dataset<ViewType, OpType>::read() {
  static_assert(std::is_same_v<OpType, specfem::io::read>,
                "read() can only be called on read datasets");

  using storage_type = typename native_type::type;

  if constexpr (std::is_same_v<value_type, bool> && !std::is_same_v<storage_type, bool>) {
    // Handle bool conversion from uint8_t
    if (std::is_same_v<MemSpace, specfem::kokkos::HostMemSpace>) {
      std::vector<storage_type> converted_data(data.size());
      DatasetBase<OpType>::read(converted_data.data(), variable);
      for (size_t i = 0; i < data.size(); ++i) {
        data.data()[i] = (converted_data[i] != 0);
      }
    } else if (std::is_same_v<MemSpace, specfem::kokkos::DevMemSpace>) {
      auto host_data = Kokkos::create_mirror_view(data);
      std::vector<storage_type> converted_data(host_data.size());
      DatasetBase<OpType>::read(converted_data.data(), variable);
      for (size_t i = 0; i < host_data.size(); ++i) {
        host_data.data()[i] = (converted_data[i] != 0);
      }
      Kokkos::deep_copy(data, host_data);
    } else {
      throw std::runtime_error("Unknown memory space");
    }
  } else {
    // Standard case - no conversion needed
    if (std::is_same_v<MemSpace, specfem::kokkos::HostMemSpace>) {
      DatasetBase<OpType>::read(data.data(), variable);
    } else if (std::is_same_v<MemSpace, specfem::kokkos::DevMemSpace>) {
      auto host_data = Kokkos::create_mirror_view(data);
      DatasetBase<OpType>::read(host_data.data(), variable);
      Kokkos::deep_copy(data, host_data);
    } else {
      throw std::runtime_error("Unknown memory space");
    }
  }
}
#endif

#endif
