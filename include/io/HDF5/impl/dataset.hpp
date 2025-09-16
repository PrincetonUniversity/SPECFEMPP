#pragma once

#ifndef NO_HDF5
#include "H5Cpp.h"
#endif

#include "datasetbase.hpp"
#include <memory>
#include <string>

namespace specfem {
namespace io {
namespace impl {
namespace HDF5 {

#ifdef NO_HDF5
// Error message if HDF5 is not available
template <typename ViewType, typename OpType> class Dataset {
public:
  using value_type = typename ViewType::non_const_value_type;
  using MemSpace = typename ViewType::memory_space;
  using native_type = void;

  template <typename... Args> Dataset(Args &&...args) {
    throw std::runtime_error("SPECFEM++ was not compiled with HDF5 support");
  }

  void write() {
    throw std::runtime_error("SPECFEM++ was not compiled with HDF5 support");
  }

  void read() {
    throw std::runtime_error("SPECFEM++ was not compiled with HDF5 support");
  }
};
#else
// Forward declaration
template <typename OpType> class Group;
template <typename OpType> class File;

/**
 * @brief Wrapper for HDF5 dataset
 *
 * @tparam ViewType Kokkos view type of the data
 * @tparam OpType Operation type (read/write)
 */
template <typename ViewType, typename OpType>
class Dataset : public DatasetBase<OpType> {
public:
  // static_assert(ViewType::is_contiguous, "ViewType must be contiguous");

  constexpr static int rank = ViewType::rank(); ///< Rank of the View

  using value_type =
      typename ViewType::non_const_value_type; ///< Underlying type of the View
  using native_type =
      typename specfem::io::impl::HDF5::native_type<value_type>; ///< Native
                                                                 ///< type used
                                                                 ///< within
                                                                 ///< HDF5
                                                                 ///< library
  using MemSpace = typename ViewType::memory_space; ///< Memory space

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Construct a new HDF5 Dataset object within an HDF5 file with the
   * given name
   *
   * @param file HDF5 file object to create the dataset in
   * @param name Name of the dataset
   * @param data Data to write
   */
  Dataset(std::unique_ptr<H5::H5File> &file, const std::string &name,
          const ViewType data);

  /**
   * @brief Construct a new HDF5 Dataset object within an HDF5 group with the
   * given name
   *
   * @param group HDF5 group object to create the dataset in
   * @param name Name of the dataset
   * @param data Data to write
   */
  Dataset(std::unique_ptr<H5::Group> &group, const std::string &name,
          const ViewType data);

  ///@}

  /**
   * @brief Write data to the dataset
   */
  void write();

  /**
   * @brief Read data from the dataset
   */
  void read();

  ~Dataset() { DatasetBase<OpType>::close(); }

private:
  ViewType data; ///< Data to be written/read
};
#endif
} // namespace HDF5
} // namespace impl
} // namespace io
} // namespace specfem
