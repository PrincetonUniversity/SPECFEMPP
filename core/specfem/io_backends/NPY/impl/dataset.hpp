#pragma once

#include "datasetbase.hpp"
#include "specfem/io_backends/impl/file_extents.hpp"
#include <boost/filesystem.hpp>
#include <string>

namespace specfem::io_backends::impl::NPY {

// Forward declaration
template <typename OpType> class Group;
template <typename OpType> class File;
/**
 * @brief Dataset class for NPY IO
 *
 * @tparam OpType Operation type (read/write)
 */
template <typename ViewType, typename OpType>
class Dataset : public DatasetBase<OpType> {
public:
  constexpr static int rank = ViewType::rank(); ///< Rank of the View

  using value_type =
      typename ViewType::non_const_value_type;      ///< Underlying type
  using MemSpace = typename ViewType::memory_space; ///< Memory space

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Construct a new npy Dataset object within an NPY file with the
   * given name
   *
   * @param file npy file object to create the dataset in
   * @param name Name of the dataset
   * @param data Data to write
   * @throws std::invalid_argument if @p data is not a contiguous LayoutLeft or
   * LayoutRight buffer
   */
  Dataset(boost::filesystem::path &file, const std::string &name,
          const ViewType data);
  ///@}

  /**
   * @brief Write the data to the dataset
   *
   */
  void write();

  /**
   * @brief Read the data from the dataset
   *
   */
  void read();

  ~Dataset() { DatasetBase<OpType>::close(); }

private:
  ViewType data; ///< Data to write
  /// Header storage order; rank 0 and 1 views are declared column-major
  bool fortran_order;
};

} // namespace specfem::io_backends::impl::NPY
