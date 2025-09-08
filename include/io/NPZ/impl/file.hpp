#pragma once

#include "enumerations/interface.hpp"
#include "group.hpp"
#include <fstream>
#include <stdexcept>
#include <string>

namespace specfem::io::impl::NPZ {

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
   * @brief Construct a new npz File object with the given name
   *
   * @param name Name of the file
   */
  File(const std::string &name)
      : file_path(name + ".npz"),
        stream(file_path, std::ios::out | std::ios::binary) {}

  /**
   * @brief Construct a new npz File object with the given name
   *
   * @param name Name of the file
   */
  File(const char *name) : File(std::string(name)) {}
  ///@}

  /**
   * @brief Create a new dataset within the file
   *
   * @tparam ViewType Kokkos view type of the data
   * @param name Name of the dataset
   * @param data Data to write
   * @return specfem::io::impl::NPZ::Dataset<ViewType, OpType> Dataset object
   */
  template <typename ViewType>
  specfem::io::impl::NPZ::Dataset<ViewType, OpType>
  createDataset(const std::string &name, const ViewType data) {
    return specfem::io::impl::NPZ::Dataset<ViewType, OpType>(*this, name, data);
  }

  /**
   * @brief Create a new group within the file
   *
   * @param name Name of the group
   * @return specfem::io::impl::NPZ::Group<OpType> Group object
   */
  specfem::io::impl::NPZ::Group<OpType> createGroup(const std::string &name) {
    return specfem::io::impl::NPZ::Group<OpType>(*this, name);
  }

  /**
   * @brief Write data to the file at the given path
   *
   * @tparam value_type Data type
   * @param data Pointer to the data buffer
   * @param path Path to the dataset within the file
   */
  template <typename value_type>
  void write(const value_type *data, const std::vector<size_t> &dims,
             const std::string &path) const {
    if (!stream.is_open()) {
      std::ostringstream oss;
      oss << "ERROR : File is closed for writing " << path;
      throw std::runtime_error(oss.str());
    }

    // Count total elements
    int total_elements = 1;
    for (int i = 0; i < dims.size(); ++i) {
      total_elements *= dims[i];
    }

    // std::string header = create_npy_header<value_type>(dims);

    // stream.write(reinterpret_cast<const char *>(&header[0]), header.size());
    // stream.write(reinterpret_cast<const char *>(data),
    //            total_elements * sizeof(value_type));
  }

  void flush() { stream.flush(); }

  ~File() { stream.close(); }

private:
  std::string file_path; ///< Path to the file
  std::ofstream stream;  ///< File stream
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
   * @brief Read the npz file with the given name
   *
   * @param name Name of the file
   */
  File(const std::string &name)
      : file_path(name + ".npz"),
        stream(file_path, std::ios::in | std::ios::binary) {
    if (!boost::filesystem::exists(file_path)) {
      throw std::runtime_error("ERROR : Folder " + name + " does not exist.");
    }
  }

  /**
   * @brief Read the npz file with the given name
   *
   * @param name Name of the file
   */
  File(const char *name) : File(std::string(name)) {}
  ///@}
  /**
   * @brief Open an existing dataset within the file
   *
   * @tparam ViewType Kokkos view type of the data
   * @param name Name of the dataset
   * @param data Data to be read
   * @return specfem::io::impl::NPZ::Dataset<ViewType, OpType> Dataset object
   */
  template <typename ViewType>
  specfem::io::impl::NPZ::Dataset<ViewType, OpType>
  openDataset(const std::string &path, const ViewType data) {
    return specfem::io::impl::NPZ::Dataset<ViewType, OpType>(*this, path, data);
  }

  /**
   * @brief Open an existing group within the file
   *
   * @param name Name of the group
   * @return specfem::io::impl::NPZ::Group<OpType> Group object
   */
  specfem::io::impl::NPZ::Group<OpType> openGroup(const std::string &name) {
    return specfem::io::impl::NPZ::Group<OpType>(*this, name);
  }

  /**
   * @brief Read data from the file at the given path
   *
   * @tparam value_type Data type
   * @param data Pointer to the data buffer
   * @param path Path to the dataset within the file
   */
  template <typename value_type>
  void read(value_type *data, const std::vector<size_t> &dims,
            const std::string &path) const {
    if (!stream.is_open()) {
      std::ostringstream oss;
      oss << "ERROR : File is closed for writing " << path;
      throw std::runtime_error(oss.str());
    }

    // Count total elements
    int total_elements = 1;
    int rank = dims.size();
    for (int i = 0; i < rank; ++i) {
      total_elements *= dims[i];
    }

    // std::vector<size_t> shape = parse_npy_header<value_type>(stream);
    // if (rank != shape.size()) {
    //   std::ostringstream oss;
    //   oss << "ERROR : Rank mismatch between dataset and file";
    //   throw std::runtime_error(oss.str());
    // }

    // for (int i = 0; i < rank; ++i) {
    //   if (dims[i] != shape[i]) {
    //     std::ostringstream oss;
    //     oss << "ERROR : Dimension mismatch between dataset and file";
    //     throw std::runtime_error(oss.str());
    //   }
    // }

    // stream.read(reinterpret_cast<char *>(data),
    //           total_elements * sizeof(value_type));
  }

  ~File() { stream.close(); }

private:
  std::string file_path; ///< Path to the file
  std::ifstream stream;  ///< File stream
};
} // namespace specfem::io::impl::NPZ
