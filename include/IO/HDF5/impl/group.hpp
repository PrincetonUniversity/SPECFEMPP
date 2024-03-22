#ifndef SPECFEM_IO_HDF5_IMPL_GROUP_HPP
#define SPECFEM_IO_HDF5_IMPL_GROUP_HPP

#include "H5Cpp.h"
#include "IO/operators.hpp"
#include "dataset.hpp"
#include <memory>
#include <string>

namespace specfem {
namespace IO {
namespace impl {
namespace HDF5 {

// Forward declaration
template <typename ViewType, typename OpType> class Dataset;
template <typename OpType> class File;

template <typename OpType> class Group;

template <> class Group<specfem::IO::write> {
public:
  using OpType = specfem::IO::write;

  Group(std::unique_ptr<H5::H5File> &file, const std::string &name)
      : group(std::make_unique<H5::Group>(file->createGroup(name))){};

  template <typename ViewType>
  specfem::IO::impl::HDF5::Dataset<ViewType, OpType>
  createDataset(const std::string &name, const ViewType data) {
    return specfem::IO::impl::HDF5::Dataset<ViewType, OpType>(group, name,
                                                              data);
  }

  ~Group() { group->close(); }

private:
  std::unique_ptr<H5::Group> group;
};

template <> class Group<specfem::IO::read> {
public:
  using OpType = specfem::IO::read;

  Group(std::unique_ptr<H5::H5File> &file, const std::string &name)
      : group(std::make_unique<H5::Group>(file->openGroup(name))){};

  template <typename ViewType>
  specfem::IO::impl::HDF5::Dataset<ViewType, OpType>
  openDataset(const std::string &name, const ViewType data) {
    return specfem::IO::impl::HDF5::Dataset<ViewType, OpType>(group, name,
                                                              data);
  }

  ~Group() { group->close(); }

private:
  std::unique_ptr<H5::Group> group;
};

} // namespace HDF5
} // namespace impl
} // namespace IO
} // namespace specfem

#endif
