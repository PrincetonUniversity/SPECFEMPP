#pragma once

#include "dataset.hpp"
#include "datasetbase.hpp"
#include <Kokkos_Core.hpp>
#include "kokkos_abstractions.h"

template <typename ViewType, typename OpType>
specfem::io::impl::NPY::Dataset<ViewType, OpType>::Dataset(
    boost::filesystem::path &folder_name, const std::string &name,
    const ViewType data)
    : data(data), DatasetBase<OpType>(
                      folder_name, name,
                      [&data]() -> std::vector<size_t> {
                        std::vector<size_t> dims;
                        for (int i = 0; i < data.rank(); i++) {
                          dims.push_back(data.extent(i));
                        }
                        return dims;
                      }()) {}

template <typename ViewType, typename OpType>
void specfem::io::impl::NPY::Dataset<ViewType, OpType>::write() {
  if (std::is_same_v<MemSpace, specfem::kokkos::HostMemSpace>) {
    DatasetBase<OpType>::write(data.data());
  } else if (std::is_same_v<MemSpace, specfem::kokkos::DevMemSpace>) {
    auto host_data = Kokkos::create_mirror_view(data);
    Kokkos::deep_copy(host_data, data);
    DatasetBase<OpType>::write(host_data.data());
    return;
  } else {
    throw std::runtime_error("Unknown memory space");
  }
}

template <typename ViewType, typename OpType>
void specfem::io::impl::NPY::Dataset<ViewType, OpType>::read() {
  if (std::is_same_v<MemSpace, specfem::kokkos::HostMemSpace>) {
    DatasetBase<OpType>::read(data.data());
  } else if (std::is_same_v<MemSpace, specfem::kokkos::DevMemSpace>) {
    auto host_data = Kokkos::create_mirror_view(data);
    DatasetBase<OpType>::read(host_data.data());
    Kokkos::deep_copy(data, host_data);
    return;
  } else {
    throw std::runtime_error("Unknown memory space");
  }
}
