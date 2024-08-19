#ifndef SPECFEM_IO_HDF5_IMPL_DATASET_TPP
#define SPECFEM_IO_HDF5_IMPL_DATASET_TPP

#include "H5Cpp.h"
#include "dataset.hpp"
#include "datasetbase.hpp"
#include "datasetbase.tpp"
#include "IO/operators.hpp"
#include "native_type.hpp"
#include "kokkos_abstractions.h"
#include <string>

// check if Kokkos version is < 4.1
#if KOKKOS_VERSION < 40100
template <typename ViewType, typename OpType>
const int specfem::IO::impl::HDF5::Dataset<ViewType, OpType>::rank = ViewType::rank;
#else
template <typename ViewType, typename OpType>
const int specfem::IO::impl::HDF5::Dataset<ViewType, OpType>::rank = ViewType::rank();
#endif

template <typename ViewType, typename OpType>
specfem::IO::impl::HDF5::Dataset<ViewType, OpType>::Dataset(
    std::unique_ptr<H5::H5File> &file, const std::string &name,
    const ViewType data)
    : data(data), DatasetBase<OpType>(file, name, rank, [&data]() -> hsize_t * {
        hsize_t *dims = new hsize_t[rank];
        for (int i = 0; i < rank; i++) {
          dims[i] = data.extent(i);
        }
        return dims;
      }(), native_type::type()) {}

template <typename ViewType, typename OpType>
specfem::IO::impl::HDF5::Dataset<ViewType, OpType>::Dataset(
    std::unique_ptr<H5::Group> &group, const std::string &name,
    const ViewType data)
    : data(data),
      DatasetBase<OpType>(group, name, rank, [&data]() -> hsize_t * {
        hsize_t *dims = new hsize_t[rank];
        for (int i = 0; i < rank; i++) {
          dims[i] = data.extent(i);
        }
        return dims;
      }(), native_type::type()) {}

template <typename ViewType, typename OpType>
void specfem::IO::impl::HDF5::Dataset<ViewType, OpType>::write() {
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
void specfem::IO::impl::HDF5::Dataset<ViewType, OpType>::read() {
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

#endif
