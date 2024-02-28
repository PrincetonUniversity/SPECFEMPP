#ifndef SPECFEM_IO_HDF5_IMPL_DATASETBASE_TPP
#define SPECFEM_IO_HDF5_IMPL_DATASETBASE_TPP

#include "H5Cpp.h"
#include "datasetbase.hpp"
#include "native_type.hpp"
#include "native_type.tpp"
#include "IO/operators.hpp"
#include <memory>
#include <string>

// template <>
// specfem::IO::impl::HDF5::DatasetBase<specfem::IO::write>::DatasetBase(
//     std::unique_ptr<H5::H5File> &file, const std::string &name,
//     const int rank, const hsize_t *dims)
//     : dataspace(std::make_unique<H5::DataSpace>(rank, dims)) {
// }

// template <>
// specfem::IO::impl::HDF5::DatasetBase<specfem::IO::write>::DatasetBase(
//     std::unique_ptr<H5::Group> &group, const std::string &name,
//     const int rank, const hsize_t *dims)
//     : dataspace(std::make_unique<H5::DataSpace>(rank, dims)) {
// }

// template <>
// specfem::IO::impl::HDF5::DatasetBase<specfem::IO::read>::DatasetBase(
//     std::unique_ptr<H5::H5File> &file, const std::string &name,
//     const int rank, const hsize_t *dims) : dataset(file->openDataSet(name)) {
//   dataspace = std::make_unique<H5::DataSpace>(dataset->getSpace());

//   hsize_t check_dims[rank];
//   dataspace->getSimpleExtentDims(check_dims);

//   for (int i = 0; i < rank; ++i) {
//     if (check_dims[i] != dims[i]) {
//       throw std::runtime_error("Dimensions of dataset do not match view");
//     }
//   }
// }

// template <>
// specfem::IO::impl::HDF5::DatasetBase<specfem::IO::read>::DatasetBase(
//     std::unique_ptr<H5::Group> &group, const std::string &name,
//     const int rank, const hsize_t *dims) : dataset(group->openDataSet(name)) {
//     dataspace = std::make_unique<H5::DataSpace>(dataset->getSpace());

//     hsize_t check_dims[rank];
//     dataspace->getSimpleExtentDims(check_dims);

//     for (int i = 0; i < rank; ++i) {
//       if (check_dims[i] != dims[i]) {
//         throw std::runtime_error("Dimensions of dataset do not match view");
//       }
//     }

// }

// template <>
// template <typename value_type>
// void specfem::IO::impl::HDF5::DatasetBase<specfem::IO::write>::write(
//     const value_type *data) {
//   this->dataset = std::make_unique<H5::DataSet>(
//       file->createDataSet(name, specfem::IO::impl::HDF5::native_type<value_type>::type(), *dataspace));
//   dataset->write(data, specfem::IO::impl::HDF5::native_type<value_type>::type());
// }

// template <>
// template <typename value_type>
// void specfem::IO::impl::HDF5::DatasetBase<specfem::IO::read>::read(
//     value_type *data) {

//   H5::DataType type = dataset->getDataType();

//   // Check that read type matches the default write type
//   assert(type.getOrder() == specfem::IO::impl::HDF5::native_type<value_type>::type().getOrder());
//   dataset->read(data, specfem::IO::impl::HDF5::native_type<value_type>::type());
// }

// template <>
// void specfem::IO::impl::HDF5::DatasetBase<specfem::IO::write>::close() {
//   dataset->close();
//   dataspace->close();
// }

#endif
