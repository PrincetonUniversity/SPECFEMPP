#ifndef SPECFEM_IO_HDF5_IMPL_FILE_HPP
#define SPECFEM_IO_HDF5_IMPL_FILE_HPP

#include "H5Cpp.h"
#include "IO/operators.hpp"
#include "dataset.hpp"
#include "group.hpp"
#include <memory>
#include <string>

namespace specfem {
namespace IO {
namespace impl {
namespace HDF5 {

// Forward declaration
template <typename OpType> class Group;
template <typename ViewType, typename OpType> class Dataset;

template <typename OpType> class File;

template <> class File<specfem::IO::write> {
public:
  using OpType = specfem::IO::write;

  File(const std::string &name)
      : file(std::make_unique<H5::H5File>(name + ".h5", H5F_ACC_TRUNC)) {}
  File(const char *name)
      : file(std::make_unique<H5::H5File>(std::string(name) + ".h5",
                                          H5F_ACC_TRUNC)) {}

  template <typename ViewType>
  specfem::IO::impl::HDF5::Dataset<ViewType, OpType>
  createDataset(const std::string &name, const ViewType data) {
    return specfem::IO::impl::HDF5::Dataset<ViewType, OpType>(file, name, data);
  }

  specfem::IO::impl::HDF5::Group<OpType> createGroup(const std::string &name) {
    return specfem::IO::impl::HDF5::Group<OpType>(file, name);
  }

  ~File() { file->close(); }

private:
  std::unique_ptr<H5::H5File> file;
};

template <> class File<specfem::IO::read> {
public:
  using OpType = specfem::IO::read;

  File(const std::string &name)
      : file(std::make_unique<H5::H5File>(name + ".h5", H5F_ACC_RDONLY)) {}
  File(const char *name)
      : file(std::make_unique<H5::H5File>(std::string(name) + ".h5",
                                          H5F_ACC_RDONLY)) {}

  template <typename ViewType>
  specfem::IO::impl::HDF5::Dataset<ViewType, OpType>
  openDataset(const std::string &name, const ViewType data) {
    return specfem::IO::impl::HDF5::Dataset<ViewType, OpType>(file, name, data);
  }

  specfem::IO::impl::HDF5::Group<OpType> openGroup(const std::string &name) {
    return specfem::IO::impl::HDF5::Group<OpType>(file, name);
  }

  ~File() { file->close(); }

private:
  std::unique_ptr<H5::H5File> file;
};
} // namespace HDF5
} // namespace impl
} // namespace IO
} // namespace specfem

#endif
