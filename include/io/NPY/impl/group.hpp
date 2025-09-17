#pragma once

#include <boost/filesystem.hpp>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace specfem::io::impl::NPY {

// Forward declaration
template <typename ViewType, typename OpType> class Dataset;
template <typename OpType> class File;

/**
 * @brief Group class for npy IO
 *
 * @tparam OpType Operation type (read/write)
 */
template <typename OpType> class Group;

/**
 * @brief Template specialization for write operation
 */
template <> class Group<specfem::io::write> {
public:
  using OpType = specfem::io::write; ///< Operation type

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Construct a new npy Group object with the given name
   *
   * @param parent_directory Path to the parent directory
   * @param name Name of the folder
   */
  Group(boost::filesystem::path parent_directory, const std::string &name)
      : folder_path(parent_directory / boost::filesystem::path(name)) {
    // Delete the folder if it exists
    if (boost::filesystem::exists(this->folder_path)) {
      std::ostringstream oss;
      oss << "WARNING : Folder " << this->folder_path.string()
          << " already exists. Deleting it.";
      std::cout << oss.str() << std::endl;
      boost::filesystem::remove_all(folder_path);
    }

    // Create the folder
    const bool success = boost::filesystem::create_directory(this->folder_path);
    if (!success) {
      std::ostringstream oss;
      oss << "ERROR : Could not create folder " << name;
      throw std::runtime_error(oss.str());
    }
  }

  /**
   * @brief Construct a new npy Group object with the given name
   *
   * @param parent_directory Path to the parent directory
   * @param name Name of the folder
   */
  Group(boost::filesystem::path parent_directory, const char *name)
      : Group(parent_directory, std::string(name)) {}
  ///@}

  /**
   * @brief Create a new dataset within the group
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
   * @brief Create a new group within the group
   *
   * @param name Name of the group
   * @return specfem::io::impl::NPY::Group<OpType> Group object
   */
  specfem::io::impl::NPY::Group<OpType> createGroup(const std::string &name) {
    return specfem::io::impl::NPY::Group<OpType>(folder_path, name);
  }

  ~Group() {}

private:
  boost::filesystem::path folder_path; ///< Path to the folder
};

/**
 * @brief Template specialization for read operation
 */
template <> class Group<specfem::io::read> {
public:
  using OpType = specfem::io::read; ///< Operation type

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Construct a new npy Group object with the given name
   *
   * @param parent_directory Path to the parent directory
   * @param name Name of the folder
   */
  Group(boost::filesystem::path parent_directory, const std::string &name)
      : folder_path(parent_directory / boost::filesystem::path(name)) {
    // Check if the folder exists
    if (!boost::filesystem::exists(this->folder_path)) {
      std::ostringstream oss;
      oss << "ERROR : Folder " << this->folder_path.string()
          << " does not exist.";
      throw std::runtime_error(oss.str());
    }
  }

  /**
   * @brief Construct a new npy Group object with the given name
   *
   * @param parent_directory Path to the parent directory
   * @param name Name of the folder
   */
  Group(boost::filesystem::path parent_directory, const char *name)
      : Group(parent_directory, std::string(name)) {}
  ///@}

  /**
   * @brief Open an existing dataset within the group
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
   * @brief Open an existing group within the group
   *
   * @param name Name of the group
   * @return specfem::io::impl::NPY::Group<OpType> Group object
   */
  specfem::io::impl::NPY::Group<OpType> openGroup(const std::string &name) {
    return specfem::io::impl::NPY::Group<OpType>(folder_path, name);
  }

  ~Group() {}

private:
  boost::filesystem::path folder_path; ///< Path to the folder
};

} // namespace specfem::io::impl::NPY
