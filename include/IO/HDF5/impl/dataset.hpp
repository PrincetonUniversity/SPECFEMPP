#ifndef SPECFEM_IO_HDF5_IMPL_DATASET_HPP
#define SPECFEM_IO_HDF5_IMPL_DATASET_HPP

#include "H5Cpp.h"
#include "file.hpp"
#include "group.hpp"
#include "native_type.hpp"
#include <memory>
#include <string>

namespace specfem {
namespace IO {
namespace impl {
namespace HDF5 {

// Forward declaration
class Group;
class File;

template <typename ViewType> class Dataset {
public:
  // static_assert(ViewType::is_contiguous, "ViewType must be contiguous");

  const static int rank;
  using value_type = typename ViewType::non_const_value_type;
  const static H5::PredType native_type;
  using MemSpace = typename ViewType::memory_space;

  Dataset(std::unique_ptr<H5::H5File> &file, const std::string &name,
          const ViewType data);
  Dataset(std::unique_ptr<H5::Group> &Group, const std::string &name,
          const ViewType data);
  void write();
  void close();

  ~Dataset() { close(); }

private:
  ViewType data;
  std::unique_ptr<H5::DataSet> dataset;
  std::unique_ptr<H5::DataSpace> dataspace;
};
} // namespace HDF5
} // namespace impl
} // namespace IO
} // namespace specfem

#endif
