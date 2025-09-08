#pragma once

#include "file.hpp"
#include <string>

namespace specfem::io::impl::NPZ {

/**
 * @brief Dataset class for NPZ IO
 *
 * @tparam OpType Operation type (read/write)
 */
template <typename ViewType, typename OpType> class Dataset {
public:
  // static_assert(ViewType::is_contiguous, "ViewType must be contiguous");

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
};

} // namespace specfem::io::impl::NPZ
