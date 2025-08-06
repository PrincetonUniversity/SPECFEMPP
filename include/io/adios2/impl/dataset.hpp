#ifndef SPECFEM_IO_ADIOS2_IMPL_DATASET_HPP
#define SPECFEM_IO_ADIOS2_IMPL_DATASET_HPP

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

#if KOKKOS_VERSION < 40100
  constexpr static int rank = ViewType::rank;
  ; ///< Rank of the View
#else
  constexpr static int rank = ViewType::rank(); ///< Rank of the View
#endif
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
   * @brief Construct a new ADIOS2 Dataset object with the given name
   *
   * @param io ADIOS2 IO object
   * @param engine ADIOS2 engine object
   * @param name Name of the dataset
   * @param data Data to write/read
   */
  Dataset(std::shared_ptr<adios2::IO> &io,
          std::shared_ptr<adios2::Engine> &engine, const std::string &name,
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
  adios2::Variable<typename native_type::type> variable; ///< ADIOS2 variable
};
#endif
} // namespace ADIOS2
} // namespace impl
} // namespace io
} // namespace specfem

#endif
