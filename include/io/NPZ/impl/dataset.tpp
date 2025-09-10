#pragma once

#include "file.hpp"
#include "dataset.hpp"
#include "enumerations/interface.hpp"
#include <Kokkos_Core.hpp>
#include "kokkos_abstractions.h"

#ifndef NO_NPZ
template <typename ViewType, typename OpType>
specfem::io::impl::NPZ::Dataset<ViewType, OpType>::Dataset(
    specfem::io::impl::NPZ::File<OpType> &file, const std::string &path,
    const ViewType data)
    : file(file), data(data), path(path), dims([&data]() -> std::vector<size_t> {
                        std::vector<size_t> dims;
                        for (int i = 0; i < data.rank(); i++) {
                          dims.push_back(data.extent(i));
                        }
                        return dims;
                      }()) {}

template <typename ViewType, typename OpType>
void specfem::io::impl::NPZ::Dataset<ViewType, OpType>::write() {
  if (std::is_same_v<MemSpace, specfem::kokkos::HostMemSpace>) {
    file.write(data.data(), dims, path);
  } else if (std::is_same_v<MemSpace, specfem::kokkos::DevMemSpace>) {
    auto host_data = Kokkos::create_mirror_view(data);
    Kokkos::deep_copy(host_data, data);
    file.write(host_data.data(), dims, path);
    return;
  } else {
    throw std::runtime_error("Unknown memory space");
  }
}

template <typename ViewType, typename OpType>
void specfem::io::impl::NPZ::Dataset<ViewType, OpType>::read() {
  if (std::is_same_v<MemSpace, specfem::kokkos::HostMemSpace>) {
    file.read(data.data(), dims, path);
  } else if (std::is_same_v<MemSpace, specfem::kokkos::DevMemSpace>) {
    auto host_data = Kokkos::create_mirror_view(data);
    file.read(host_data.data(), dims, path);
    Kokkos::deep_copy(data, host_data);
    return;
  } else {
    throw std::runtime_error("Unknown memory space");
  }
}
#endif
