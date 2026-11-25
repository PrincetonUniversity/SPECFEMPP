#pragma once

#include "file.hpp"
#include "specfem/io_backends/impl/file_extents.hpp"
#include <string>

namespace specfem::io_backends::impl::NPZ {

#ifdef NO_NPZ
// Error message if NPZ is not available
template <typename ViewType, typename OpType> class Dataset {
public:
  using value_type = typename ViewType::non_const_value_type;
  using MemSpace = typename ViewType::memory_space;
  using native_type = void;

  template <typename... Args> Dataset(Args &&...args) {
    throw std::runtime_error("SPECFEM++ was not compiled with NPZ support");
  }

  void write() {
    throw std::runtime_error("SPECFEM++ was not compiled with NPZ support");
  }

  void read() {
    throw std::runtime_error("SPECFEM++ was not compiled with NPZ support");
  }
};
#else
/**
 * @brief Dataset class for NPZ IO
 *
 * @tparam OpType Operation type (read/write)
 */
template <typename ViewType, typename OpType> class Dataset {
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
   * @brief Construct a new npz Dataset object within an NPZ file with the
   * given name
   *
   * @param stream input or output file stream
   * @param path Name of the dataset
   * @param data Data to write
   * @throws std::invalid_argument if @p data is not a contiguous LayoutLeft or
   * LayoutRight buffer
   */
  Dataset(File<OpType> &file, const std::string &path, const ViewType data);
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

  /**
   * @brief Close the dataset
   *
   */
  void close() const {}

private:
  ViewType data; ///< Data to write
  File<OpType> &file;
  const std::string path;
  const std::vector<size_t> dims;
  /// Header storage order; rank 0 and 1 views are declared column-major
  const bool fortran_order;
};
#endif
} // namespace specfem::io_backends::impl::NPZ
