#ifndef _SPECFEM_IO_ASCII_IMPL_DATASET_HPP
#define _SPECFEM_IO_ASCII_IMPL_DATASET_HPP

#include "datasetbase.hpp"
#include "native_type.hpp"
#include <boost/filesystem.hpp>
#include <string>

namespace specfem {
namespace IO {
namespace impl {
namespace ASCII {

// Forward declaration
template <typename OpType> class Group;
template <typename OpType> class File;

template <typename ViewType, typename OpType>
class Dataset : public DatasetBase<OpType> {
public:
  // static_assert(ViewType::is_contiguous, "ViewType must be contiguous");

  const static int rank;
  using value_type = typename ViewType::non_const_value_type;
  using native_type =
      typename specfem::IO::impl::ASCII::native_type<value_type>;
  using MemSpace = typename ViewType::memory_space;

  Dataset(boost::filesystem::path &file, const std::string &name,
          const ViewType data);

  void write();

  void read();

  ~Dataset() { DatasetBase<OpType>::close(); }

private:
  ViewType data;
};
} // namespace ASCII
} // namespace impl
} // namespace IO
} // namespace specfem

#endif /* _SPECFEM_IO_ASCII_IMPL_DATASET_HPP */
