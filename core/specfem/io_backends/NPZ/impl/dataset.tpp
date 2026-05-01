#pragma once

#include "dataset.hpp"
#include "specfem/enums.hpp"
#include "file.hpp"

#include <Kokkos_Core.hpp>

#ifndef NO_NPZ
template <typename ViewType, typename OpType>
specfem::io_backends::impl::NPZ::Dataset<ViewType, OpType>::Dataset(
    specfem::io_backends::impl::NPZ::File<OpType> &file, const std::string &path,
    const ViewType data)
    : file(file), data(data), path(path),
      dims([&data]() -> std::vector<size_t> {
        std::vector<size_t> dims;
        for (int i = 0; i < data.rank(); i++) {
          dims.push_back(data.extent(i));
        }
        return dims;
      }()) {}

template <typename ViewType, typename OpType>
void specfem::io_backends::impl::NPZ::Dataset<ViewType, OpType>::write() {
  if (std::is_same_v<MemSpace, Kokkos::HostSpace>) {
    file.write(data.data(), dims, path);
  } else if (std::is_same_v<MemSpace,
                            Kokkos::DefaultExecutionSpace::memory_space>) {
    auto host_data = Kokkos::create_mirror_view(data);
    Kokkos::deep_copy(host_data, data);
    file.write(host_data.data(), dims, path);
    return;
  } else {
    throw std::runtime_error("Unknown memory space");
  }
}

template <typename ViewType, typename OpType>
void specfem::io_backends::impl::NPZ::Dataset<ViewType, OpType>::read() {
  if (std::is_same_v<MemSpace, Kokkos::HostSpace>) {
    file.read(data.data(), dims, path);
  } else if (std::is_same_v<MemSpace,
                            Kokkos::DefaultExecutionSpace::memory_space>) {
    auto host_data = Kokkos::create_mirror_view(data);
    file.read(host_data.data(), dims, path);
    Kokkos::deep_copy(data, host_data);
    return;
  } else {
    throw std::runtime_error("Unknown memory space");
  }
}
#endif
