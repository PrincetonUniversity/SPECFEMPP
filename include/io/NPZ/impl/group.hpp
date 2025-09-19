#pragma once

#include "enumerations/interface.hpp"
#include <stdexcept>
#include <string>

namespace specfem::io::impl::NPZ {

// Forward declaration
template <typename ViewType, typename OpType> class Dataset;
template <typename OpType> class File;

#ifdef NO_NPZ
template <typename OpType> class Group {
public:
  template <typename... Args> Group(Args &&...args) {
    throw std::runtime_error("SPECFEM++ was not compiled with NPZ support");
  }

  template <typename ViewType>
  specfem::io::impl::NPZ::Dataset<ViewType, OpType>
  createDataset(const std::string &name, const ViewType data) {
    throw std::runtime_error("SPECFEM++ was not compiled with NPZ support");
  }

  specfem::io::impl::NPZ::Group<OpType> createGroup(const std::string &name) {
    throw std::runtime_error("SPECFEM++ was not compiled with NPZ support");
  }

  template <typename ViewType>
  specfem::io::impl::NPZ::Dataset<ViewType, OpType>
  openDataset(const std::string &name, const ViewType data) {
    throw std::runtime_error("SPECFEM++ was not compiled with NPZ support");
  }

  specfem::io::impl::NPZ::Group<OpType> openGroup(const std::string &name) {
    throw std::runtime_error("SPECFEM++ was not compiled with NPZ support");
  }
};

#else

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
   * @param file File object
   * @param name Name of the folder
   */
  Group(File<OpType> &file, const std::string &name)
      : file(file), group_name(name) {}

  /**
   * @brief Construct a new npy Group object with the given name
   *
   * @param file File object
   * @param parent_group_name Path to the parent directory
   * @param name Name of the folder
   */
  Group(File<OpType> &file, const std::string parent_group_name,
        const std::string &name)
      : file(file), group_name(parent_group_name + "/" + name) {}
  ///@}

  /**
   * @brief Create a new dataset within the group
   *
   * @tparam ViewType Kokkos view type of the data
   * @param name Name of the dataset
   * @param data Data to write
   * @return specfem::io::impl::NPZ::Dataset<ViewType, OpType> Dataset object
   */
  template <typename ViewType>
  specfem::io::impl::NPZ::Dataset<ViewType, OpType>
  createDataset(const std::string &name, const ViewType data) {
    return specfem::io::impl::NPZ::Dataset<ViewType, OpType>(
        file, group_name + "/" + name + ".npy", data);
  }

  /**
   * @brief Create a new group within the group
   *
   * @param name Name of the group
   * @return specfem::io::impl::NPZ::Group<OpType> Group object
   */
  specfem::io::impl::NPZ::Group<OpType> createGroup(const std::string &name) {
    return specfem::io::impl::NPZ::Group<OpType>(file, group_name, name);
  }

  ~Group() {}

private:
  File<OpType> &file;
  std::string group_name; ///< Path to the folder
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
   * @param file File object
   * @param name Name of the folder
   */
  Group(File<OpType> &file, const std::string &name)
      : file(file), group_name(name) {}

  /**
   * @brief Construct a new npy Group object with the given name
   *
   * @param file File object
   * @param parent_group_name Path to the parent directory
   * @param name Name of the folder
   */
  Group(File<OpType> &file, const std::string parent_group_name,
        const std::string &name)
      : file(file), group_name(parent_group_name + "/" + name) {}
  ///@}

  /**
   * @brief Open an existing dataset within the group
   *
   * @tparam ViewType Kokkos view type of the data
   * @param name Name of the dataset
   * @param data Data to be read
   * @return specfem::io::impl::NPZ::Dataset<ViewType, OpType> Dataset object
   */
  template <typename ViewType>
  specfem::io::impl::NPZ::Dataset<ViewType, OpType>
  openDataset(const std::string &name, const ViewType data) {
    return specfem::io::impl::NPZ::Dataset<ViewType, OpType>(
        file, group_name + "/" + name + ".npy", data);
  }

  /**
   * @brief Open an existing group within the group
   *
   * @param name Name of the group
   * @return specfem::io::impl::NPZ::Group<OpType> Group object
   */
  specfem::io::impl::NPZ::Group<OpType> openGroup(const std::string &name) {
    return specfem::io::impl::NPZ::Group<OpType>(file, group_name, name);
  }

  ~Group() {}

private:
  File<OpType> &file;
  std::string group_name; ///< Path to the folder
};

#endif

} // namespace specfem::io::impl::NPZ
