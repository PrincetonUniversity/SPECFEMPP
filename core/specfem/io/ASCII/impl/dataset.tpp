#ifndef _SPECFEM_IO_ASCII_IMPL_DATASET_TPP
#define _SPECFEM_IO_ASCII_IMPL_DATASET_TPP

#include "dataset.hpp"
#include "datasetbase.hpp"
#include <Kokkos_Core.hpp>
#include "kokkos_abstractions.h"

template <typename ViewType, typename OpType>
specfem::io::impl::ASCII::Dataset<ViewType, OpType>::Dataset(
    boost::filesystem::path &folder_name, const std::string &name,
    const ViewType data)
    : data(data), DatasetBase<OpType>(
                      folder_name, name, rank,
                      [&data]() -> std::vector<int> {
                        std::vector<int> dims(rank);
                        for (int i = 0; i < rank; i++) {
                          dims[i] = data.extent(i);
                        }
                        return dims;
                      }()) {}

template <typename ViewType, typename OpType>
void specfem::io::impl::ASCII::Dataset<ViewType, OpType>::write() {
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
void specfem::io::impl::ASCII::Dataset<ViewType, OpType>::read() {
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

#endif /* _SPECFEM_IO_ASCII_IMPL_DATASET_TPP */
