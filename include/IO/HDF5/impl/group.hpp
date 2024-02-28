#ifndef SPECFEM_IO_HDF5_IMPL_GROUP_HPP
#define SPECFEM_IO_HDF5_IMPL_GROUP_HPP

#include "H5Cpp.h"
#include "dataset.hpp"
#include <memory>
#include <string>

namespace specfem {
namespace IO {
namespace impl {
namespace HDF5 {

// Forward declaration
template <typename ViewType> class Dataset;
class File;

class Group {
public:
  Group(std::unique_ptr<H5::H5File> &file, const std::string &name)
      : group(std::make_unique<H5::Group>(file->createGroup(name))) {}

  template <typename ViewType>
  specfem::IO::impl::HDF5::Dataset<ViewType>
  createDataset(const std::string &name, const ViewType data) {
    return specfem::IO::impl::HDF5::Dataset<ViewType>(group, name, data);
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
