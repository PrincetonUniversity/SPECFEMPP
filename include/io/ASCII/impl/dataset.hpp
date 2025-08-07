#pragma once

#include "datasetbase.hpp"
#include "native_type.hpp"
#include <boost/filesystem.hpp>
#include <string>

namespace specfem::io::impl::ASCII {

// Forward declaration
template <typename OpType> class Group;
template <typename OpType> class File;
/**
 * @brief Dataset class for ASCII IO
 *
 * @tparam OpType Operation type (read/write)
 */
template <typename ViewType, typename OpType>
class Dataset : public DatasetBase<OpType> {
public:
  // static_assert(ViewType::is_contiguous, "ViewType must be contiguous");

  constexpr static int rank = ViewType::rank(); ///< Rank of the View

  using value_type =
      typename ViewType::non_const_value_type; ///< Underlying type
  using native_type =
      typename specfem::io::impl::ASCII::native_type<value_type>; ///< Native
                                                                  ///< type used
                                                                  ///< within
                                                                  ///< ASCII
                                                                  ///< library
  using MemSpace = typename ViewType::memory_space; ///< Memory space

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Construct a new ASCII Dataset object within an ASCII file with the
   * given name
   *
   * @param file ASCII file object to create the dataset in
   * @param name Name of the dataset
   * @param data Data to write
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
};

} // namespace specfem::io::impl::ASCII
