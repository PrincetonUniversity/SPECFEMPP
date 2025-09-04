#pragma once

#ifndef NO_HDF5
#include "H5Cpp.h"
#endif

#include "io/operators.hpp"
#include "native_type.tpp"
#include <memory>
#include <string>

namespace specfem {
namespace io {
namespace impl {
namespace HDF5 {

#ifndef NO_HDF5
template <typename OpType> class DatasetBase;

template <> class DatasetBase<specfem::io::write> {
protected:
  template <typename AtomType>
  DatasetBase(std::unique_ptr<H5::H5File> &file, const std::string &name,
              const int rank, const hsize_t *dims, const AtomType &type)
      : dataspace(std::make_unique<H5::DataSpace>(rank, dims)) {
    dataset = std::make_unique<H5::DataSet>(
        file->createDataSet(name, type, *dataspace));
  }

  template <typename AtomType>
  DatasetBase(std::unique_ptr<H5::Group> &group, const std::string &name,
              const int rank, const hsize_t *dims, const AtomType &type)
      : dataspace(std::make_unique<H5::DataSpace>(rank, dims)) {
    dataset = std::make_unique<H5::DataSet>(
        group->createDataSet(name, type, *dataspace));
  }

  template <typename value_type> void write(const value_type *data) {
    dataset->write(data,
                   specfem::io::impl::HDF5::native_type<value_type>::type());
  }

  void write(const std::string *data) {
    hsize_t num_elements = dataspace->getSimpleExtentNpoints();
    std::vector<const char *> cstrs(num_elements);
    for (size_t i = 0; i < cstrs.size(); ++i)
      cstrs[i] = data[i].c_str();
    dataset->write(cstrs.data(),
                   specfem::io::impl::HDF5::native_type<std::string>::type());
  }

  void close() {
    dataset->close();
    dataspace->close();
  }

  ~DatasetBase() { close(); }

private:
  std::unique_ptr<H5::DataSet> dataset;
  std::unique_ptr<H5::DataSpace> dataspace;
};

template <> class DatasetBase<specfem::io::read> {
protected:
  template <typename AtomType>
  DatasetBase(std::unique_ptr<H5::H5File> &file, const std::string &name,
              const int rank, const hsize_t *dims, const AtomType &type)
      : dataset(std::make_unique<H5::DataSet>(file->openDataSet(name))) {

    if (!(dataset->getDataType() == type)) {
      throw std::runtime_error("Type of dataset does not match view");
    }

    dataspace = std::make_unique<H5::DataSpace>(dataset->getSpace());
    hsize_t check_dims[rank];
    dataspace->getSimpleExtentDims(check_dims);
    for (int i = 0; i < rank; ++i) {
      if (check_dims[i] != dims[i]) {
        throw std::runtime_error("Dimensions of dataset do not match view");
      }
    }
  }

  template <typename AtomType>
  DatasetBase(std::unique_ptr<H5::Group> &group, const std::string &name,
              const int rank, const hsize_t *dims, const AtomType &type)
      : dataset(std::make_unique<H5::DataSet>(group->openDataSet(name))) {
    if (!(dataset->getDataType() == type)) {
      throw std::runtime_error("Type of dataset does not match view");
    }

    dataspace = std::make_unique<H5::DataSpace>(dataset->getSpace());
    hsize_t check_dims[rank];
    dataspace->getSimpleExtentDims(check_dims);
    for (int i = 0; i < rank; ++i) {
      if (check_dims[i] != dims[i]) {
        throw std::runtime_error("Dimensions of dataset do not match view");
      }
    }
  }

  template <typename value_type> void read(value_type *data) {
    dataset->read(data,
                  specfem::io::impl::HDF5::native_type<value_type>::type());
  }

  void read(std::string *data) {
    hsize_t num_elements = dataspace->getSimpleExtentNpoints();
    char **buffer = new char *[num_elements];
    dataset->read(buffer,
                  specfem::io::impl::HDF5::native_type<std::string>::type());
    for (int i = 0; i < num_elements; i++) {
      data[i].assign(buffer[i]);
    }
    delete[] buffer;
  }

  void close() {
    dataset->close();
    dataspace->close();
  }

  ~DatasetBase() { close(); }

private:
  std::unique_ptr<H5::DataSet> dataset;
  std::unique_ptr<H5::DataSpace> dataspace;
};
#endif
} // namespace HDF5
} // namespace impl
} // namespace io
} // namespace specfem
