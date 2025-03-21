#ifndef SPECFEM_IO_HDF5_IMPL_FILE_HPP
#define SPECFEM_IO_HDF5_IMPL_FILE_HPP

#ifndef NO_HDF5
#include "H5Cpp.h"
#endif

#include "IO/operators.hpp"
#include "dataset.hpp"
#include "group.hpp"
#include <memory>
#include <string>

namespace specfem {
namespace io {
namespace impl {
namespace HDF5 {

#ifdef NO_HDF5
template <typename OpType> class File {
public:
  File(const std::string &name) {
    throw std::runtime_error("SPECFEM++ was not compiled with HDF5 support");
  }
  File(const char *name) {
    throw std::runtime_error("SPECFEM++ was not compiled with HDF5 support");
  }

  template <typename ViewType>
  specfem::io::impl::HDF5::Dataset<ViewType, OpType>
  createDataset(const std::string &name, const ViewType data) {
    throw std::runtime_error("SPECFEM++ was not compiled with HDF5 support");
  }

  specfem::io::impl::HDF5::Group<OpType> createGroup(const std::string &name) {
    throw std::runtime_error("SPECFEM++ was not compiled with HDF5 support");
  }

  template <typename ViewType>
  specfem::io::impl::HDF5::Dataset<ViewType, OpType>
  openDataset(const std::string &name, const ViewType data) {
    throw std::runtime_error("SPECFEM++ was not compiled with HDF5 support");
  }

  specfem::io::impl::HDF5::Group<OpType> openGroup(const std::string &name) {
    throw std::runtime_error("SPECFEM++ was not compiled with HDF5 support");
  }
};

#else

// Forward declaration
template <typename OpType> class Group;
template <typename ViewType, typename OpType> class Dataset;

/**
 * @brief Wrapper for HDF5 file
 *
 * @tparam OpType Operation type (read/write)
 */
template <typename OpType> class File;

/**
 * @brief Template specialization for write operation
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
   * @brief Construct a new HDF5 File object with the given name
   *
   * @param name Name of the file
   */
  File(const std::string &name)
      : file(std::make_unique<H5::H5File>(name + ".h5", H5F_ACC_TRUNC)) {}
  /**
   * @brief Construct a new File object with the given name
   *
   * @param name Name of the file
   */
  File(const char *name)
      : file(std::make_unique<H5::H5File>(std::string(name) + ".h5",
                                          H5F_ACC_TRUNC)) {}
  ///@}

  /**
   * @brief Create a new dataset within the file
   *
   * @tparam ViewType Kokkos view type of the data
   * @param name Name of the dataset
   * @param data Data to be written
   * @return specfem::io::impl::HDF5::Dataset<ViewType, OpType> Dataset object
   */
  template <typename ViewType>
  specfem::io::impl::HDF5::Dataset<ViewType, OpType>
  createDataset(const std::string &name, const ViewType data) {
    return specfem::io::impl::HDF5::Dataset<ViewType, OpType>(file, name, data);
  }

  /**
   * @brief Create a new group within the file
   *
   * @param name Name of the group
   * @return specfem::io::impl::HDF5::Group<OpType> Group object
   */
  specfem::io::impl::HDF5::Group<OpType> createGroup(const std::string &name) {
    return specfem::io::impl::HDF5::Group<OpType>(file, name);
  }

  ~File() { file->close(); }

private:
  std::unique_ptr<H5::H5File> file; ///< pointer to HDF5 file object
};

/**
 * @brief Template specialization for read operation
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
   * @brief Read the HDF5 file with the given name
   *
   * @param name Name of the file
   */
  File(const std::string &name)
      : file(std::make_unique<H5::H5File>(name + ".h5", H5F_ACC_RDONLY)) {}
  /**
   * @brief Read the HDF5 file with the given name
   *
   * @param name Name of the file
   */
  File(const char *name)
      : file(std::make_unique<H5::H5File>(std::string(name) + ".h5",
                                          H5F_ACC_RDONLY)) {}

  ///@}

  /**
   * @brief Open an existing dataset within the file
   *
   * @tparam ViewType Kokkos view type of the data
   * @param name Name of the dataset
   * @param data Data to be read
   * @return specfem::io::impl::HDF5::Dataset<ViewType, OpType> Dataset object
   */
  template <typename ViewType>
  specfem::io::impl::HDF5::Dataset<ViewType, OpType>
  openDataset(const std::string &name, const ViewType data) {
    return specfem::io::impl::HDF5::Dataset<ViewType, OpType>(file, name, data);
  }

  /**
   * @brief Open an existing group within the file
   *
   * @param name Name of the group
   * @return specfem::io::impl::HDF5::Group<OpType> Group object
   */
  specfem::io::impl::HDF5::Group<OpType> openGroup(const std::string &name) {
    return specfem::io::impl::HDF5::Group<OpType>(file, name);
  }

  ~File() { file->close(); }

private:
  std::unique_ptr<H5::H5File> file; ///< pointer to HDF5 file object
};

#endif

} // namespace HDF5
} // namespace impl
} // namespace io
} // namespace specfem

#endif
