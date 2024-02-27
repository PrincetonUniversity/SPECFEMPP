#ifndef SPECFEM_IO_HDF5_IMPL_DATASET_TPP
#define SPECFEM_IO_HDF5_IMPL_DATASET_TPP

#include "H5Cpp.h"
#include "dataset.hpp"
#include "native_type.hpp"
#include <string>

template <typename ViewType>
const H5::PredType specfem::IO::impl::HDF5::Dataset<ViewType>::native_type =
    specfem::IO::impl::HDF5::native_type<
        typename ViewType::non_const_value_type>::type;

// check if Kokkos version is < 4.1
#if KOKKOS_VERSION < 40100
template <typename ViewType>
const int specfem::IO::impl::HDF5::Dataset<ViewType>::rank = ViewType::rank;
#else
template <typename ViewType>
const int specfem::IO::impl::HDF5::Dataset<ViewType>::rank = ViewType::rank();
#endif

template <typename ViewType>
specfem::IO::impl::HDF5::Dataset<ViewType>::Dataset(
    std::unique_ptr<H5::H5File> &file, const std::string &name,
    const ViewType data)
    : data(data), dataspace(std::make_unique<H5::DataSpace>(
                      rank,
                      [&data]() -> hsize_t * {
                        hsize_t *dims = new hsize_t[rank];
                        for (int i = 0; i < rank; ++i) {
                          dims[i] = data.extent(i);
                        }
                        return dims;
                      }())),
      dataset(std::make_unique<H5::DataSet>(
          file->createDataSet(name, native_type, *dataspace))) {
  ASSERT(data.span_is_contiguous() == true, "ViewType must be contiguous");
}

template <typename ViewType>
specfem::IO::impl::HDF5::Dataset<ViewType>::Dataset(
    std::unique_ptr<H5::Group> &group, const std::string &name,
    const ViewType data)
    : data(data), dataspace(std::make_unique<H5::DataSpace>(
                      rank,
                      [&data]() -> hsize_t * {
                        hsize_t *dims = new hsize_t[rank];
                        for (int i = 0; i < rank; ++i) {
                          dims[i] = data.extent(i);
                        }
                        return dims;
                      }())),
      dataset(std::make_unique<H5::DataSet>(
          group->createDataSet(name, native_type, *dataspace))) {

  ASSERT(data.span_is_contiguous() == true, "ViewType must be contiguous");
}

template <typename ViewType>
void specfem::IO::impl::HDF5::Dataset<ViewType>::write() {
  if (std::is_same_v<MemSpace, specfem::kokkos::HostMemSpace>) {
    dataset->write(data.data(), native_type);
    return;
  } else if (std::is_same_v<MemSpace, specfem::kokkos::DevMemSpace>) {
    auto host_data = Kokkos::create_mirror_view(data);
    Kokkos::deep_copy(host_data, data);
    dataset->write(host_data.data(), native_type);
    return;
  } else {
    throw std::runtime_error("Unknown memory space");
  }
}

template <typename ViewType>
void specfem::IO::impl::HDF5::Dataset<ViewType>::close() {
  dataset->close();
  dataspace->close();
}

#endif
