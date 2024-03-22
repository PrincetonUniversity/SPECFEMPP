#ifndef SPECFEM_IO_HDF5_IMPL_DATASET_HPP
#define SPECFEM_IO_HDF5_IMPL_DATASET_HPP

#include "H5Cpp.h"
#include "IO/operators.hpp"
#include "datasetbase.hpp"
#include <memory>
#include <string>

namespace specfem {
namespace IO {
namespace impl {
namespace HDF5 {

// Forward declaration
template <typename OpType> class Group;
template <typename OpType> class File;

template <typename ViewType, typename OpType>
class Dataset : public DatasetBase<OpType> {
public:
  // static_assert(ViewType::is_contiguous, "ViewType must be contiguous");

  const static int rank;
  using value_type = typename ViewType::non_const_value_type;
  using native_type = typename specfem::IO::impl::HDF5::native_type<value_type>;
  using MemSpace = typename ViewType::memory_space;

  Dataset(std::unique_ptr<H5::H5File> &file, const std::string &name,
          const ViewType data);
  Dataset(std::unique_ptr<H5::Group> &group, const std::string &name,
          const ViewType data);

  void write();

  void read();

  ~Dataset() { DatasetBase<OpType>::close(); }

private:
  ViewType data;
};
} // namespace HDF5
} // namespace impl
} // namespace IO
} // namespace specfem

#endif
