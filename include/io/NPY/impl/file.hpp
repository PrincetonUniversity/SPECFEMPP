#pragma once

#include "group.hpp"
#include <boost/filesystem.hpp>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace specfem::io::impl::NPY {

// Forward declaration
template <typename OpType> class Group;
template <typename ViewType, typename OpType> class Dataset;

/**
 * @brief Numpy File implementation
 *
 *
 * @tparam OpType Operation type (read/write)
 */
template <typename OpType> class File;

/**
 * @brief Template specialization for write operation
 *
 */
template <> class File<specfem::io::write> {
public:
  using OpType = specfem::io::write; ///< Operation type

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Construct a new npy File object with the given name
   *
   * @param name Name of the folder
   */
  File(const std::string &name) : folder_path(name) {
    // Delete the folder if it exists
    if (boost::filesystem::exists(folder_path)) {
      std::ostringstream oss;
      oss << "WARNING : Folder " << folder_path.string()
          << " already exists. Deleting it.";
      std::cout << oss.str() << std::endl;
      boost::filesystem::remove_all(folder_path);
    }

    // Create the folder
    const bool success = boost::filesystem::create_directory(folder_path);
    if (!success) {
      std::ostringstream oss;
      oss << "ERROR : Could not create folder " << name;
      throw std::runtime_error(oss.str());
    }
  }

  /**
   * @brief Construct a new npy File object with the given name
   *
   * @param name Name of the folder
   */
  File(const char *name) : File(std::string(name)) {}
  ///@}

  /**
   * @brief Create a new dataset within the file
   *
   * @tparam ViewType Kokkos view type of the data
   * @param name Name of the dataset
   * @param data Data to write
   * @return specfem::io::impl::NPY::Dataset<ViewType, OpType> Dataset object
   */
  template <typename ViewType>
  specfem::io::impl::NPY::Dataset<ViewType, OpType>
  createDataset(const std::string &name, const ViewType data) {
    return specfem::io::impl::NPY::Dataset<ViewType, OpType>(folder_path, name,
                                                             data);
  }

  /**
   * @brief Create a new group within the file
   *
   * @param name Name of the group
   * @return specfem::io::impl::NPY::Group<OpType> Group object
   */
  specfem::io::impl::NPY::Group<OpType> createGroup(const std::string &name) {
    return specfem::io::impl::NPY::Group<OpType>(folder_path, name);
  }

  void flush() {};

  ~File() {}

private:
  boost::filesystem::path folder_path; ///< Path to the folder
};

/**
 * @brief Template specialization for read operation
 *
 */
template <> class File<specfem::io::read> {
public:
  using OpType = specfem::io::read; ///< Operation type

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Read the npy file with the given name
   *
   * @param name Name of the folder
   */
  File(const std::string &name) : folder_path(name) {
    if (!boost::filesystem::exists(folder_path)) {
      throw std::runtime_error("ERROR : Folder " + name + " does not exist.");
    }
  }

  /**
   * @brief Read the npy file with the given name
   *
   * @param name Name of the folder
   */
  File(const char *name) : File(std::string(name)) {}
  ///@}
  /**
   * @brief Open an existing dataset within the file
   *
   * @tparam ViewType Kokkos view type of the data
   * @param name Name of the dataset
   * @param data Data to be read
   * @return specfem::io::impl::NPY::Dataset<ViewType, OpType> Dataset object
   */
  template <typename ViewType>
  specfem::io::impl::NPY::Dataset<ViewType, OpType>
  openDataset(const std::string &name, const ViewType data) {
    return specfem::io::impl::NPY::Dataset<ViewType, OpType>(folder_path, name,
                                                             data);
  }

  /**
   * @brief Open an existing group within the file
   *
   * @param name Name of the group
   * @return specfem::io::impl::NPY::Group<OpType> Group object
   */
  specfem::io::impl::NPY::Group<OpType> openGroup(const std::string &name) {
    return specfem::io::impl::NPY::Group<OpType>(folder_path, name);
  }

  ~File() {}

private:
  boost::filesystem::path folder_path; ///< Path to the folder
};
} // namespace specfem::io::impl::NPY
