#pragma once

#ifndef NO_ADIOS2
#include <adios2.h>
#endif

#include "datasetbase.hpp"
#include "native_type.hpp"
#include <memory>
#include <string>

namespace specfem {
namespace io {
namespace impl {
namespace ADIOS2 {

#ifdef NO_ADIOS2
// Error message if ADIOS2 is not available
template <typename ViewType, typename OpType> class Dataset {
public:
  using value_type = typename ViewType::non_const_value_type;
  using MemSpace = typename ViewType::memory_space;
  using native_type = void;

  template <typename... Args> Dataset(Args &&...args) {
    throw std::runtime_error("SPECFEM++ was not compiled with ADIOS2 support");
  }

  void write() {
    throw std::runtime_error("SPECFEM++ was not compiled with ADIOS2 support");
  }

  void read() {
    throw std::runtime_error("SPECFEM++ was not compiled with ADIOS2 support");
  }
};
#else
// Forward declaration
template <typename OpType> class Group;
template <typename OpType> class File;

/**
 * @brief Wrapper for ADIOS2 dataset
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
      typename specfem::io::impl::ADIOS2::native_type<value_type>; ///< Native
                                                                   ///< type
                                                                   ///< used
                                                                   ///< within
                                                                   ///< ADIOS2
                                                                   ///< library
  using MemSpace = typename ViewType::memory_space; ///< Memory space

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Construct a new ADIOS2 Dataset object for writing (only available
   * for write operations)
   *
   * @param io ADIOS2 IO object
   * @param engine ADIOS2 engine object
   * @param name Name of the dataset
   * @param data Data to write
   */
  template <typename T = OpType>
  Dataset(std::shared_ptr<adios2::IO> &io,
          std::shared_ptr<adios2::Engine> &engine, const std::string &name,
          const ViewType data,
          std::enable_if_t<std::is_same_v<T, specfem::io::write>, int> = 0);

  /**
   * @brief Construct a new ADIOS2 Dataset object for reading (only available
   * for read operations)
   *
   * @param io ADIOS2 IO object
   * @param engine ADIOS2 engine object
   * @param name Name of the dataset
   * @param data Data to read into
   */
  template <typename T = OpType>
  Dataset(std::shared_ptr<adios2::IO> &io,
          std::shared_ptr<adios2::Engine> &engine, const std::string &name,
          const ViewType data,
          std::enable_if_t<std::is_same_v<T, specfem::io::read>, int> = 0);

  ///@}

  /**
   * @brief Write data to the dataset (only available for write operations)
   */
  template <typename T = OpType>
  std::enable_if_t<std::is_same_v<T, specfem::io::write>, void> write();

  /**
   * @brief Read data from the dataset (only available for read operations)
   */
  template <typename T = OpType>
  std::enable_if_t<std::is_same_v<T, specfem::io::read>, void> read();

  ~Dataset() { DatasetBase<OpType>::close(); }

private:
  /**
   * @brief Convert Kokkos view dimensions to ADIOS2 format
   * @param data The Kokkos view
   * @return Tuple of (shape, start, count) vectors
   */
  auto convert_dimensions(const ViewType &data) {
    std::vector<std::size_t> shape(rank);
    std::vector<std::size_t> start(rank, 0);
    std::vector<std::size_t> count(rank);

    for (int i = 0; i < rank; ++i) {
      shape[i] = data.extent(i);
      count[i] = data.extent(i);
    }

    return std::make_tuple(shape, start, count);
  }

  ViewType data; ///< Data to be written/read
  adios2::Variable<decltype(native_type::type())> variable; ///< ADIOS2 variable
};
#endif
} // namespace ADIOS2
} // namespace impl
} // namespace io
} // namespace specfem
