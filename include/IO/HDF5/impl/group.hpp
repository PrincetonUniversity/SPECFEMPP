#ifndef SPECFEM_IO_HDF5_IMPL_GROUP_HPP
#define SPECFEM_IO_HDF5_IMPL_GROUP_HPP

#ifndef NO_HDF5
#include "H5Cpp.h"
#endif

#include "IO/operators.hpp"
#include "dataset.hpp"
#include <memory>
#include <string>

namespace specfem {
namespace IO {
namespace impl {
namespace HDF5 {

#ifdef NO_HDF5
template <typename OpType> class Group {
public:
  Group(std::unique_ptr<H5::H5File> &file, const std::string &name) {
    throw std::runtime_error("SPECFEM++ was not compiled with HDF5 support");
  }
  Group(std::unique_ptr<H5::Group> &group, const std::string &name) {
    throw std::runtime_error("SPECFEM++ was not compiled with HDF5 support");
  }

  template <typename ViewType>
  specfem::IO::impl::HDF5::Dataset<ViewType, OpType>
  createDataset(const std::string &name, const ViewType data) {
    throw std::runtime_error("SPECFEM++ was not compiled with HDF5 support");
  }

  specfem::IO::impl::HDF5::Group<OpType> createGroup(const std::string &name) {
    throw std::runtime_error("SPECFEM++ was not compiled with HDF5 support");
  }

  template <typename ViewType>
  specfem::IO::impl::HDF5::Dataset<ViewType, OpType>
  openDataset(const std::string &name, const ViewType data) {
    throw std::runtime_error("SPECFEM++ was not compiled with HDF5 support");
  }

  specfem::IO::impl::HDF5::Group<OpType> openGroup(const std::string &name) {
    throw std::runtime_error("SPECFEM++ was not compiled with HDF5 support");
  }
};

#else

// Forward declaration
template <typename ViewType, typename OpType> class Dataset;
template <typename OpType> class File;

/**
 * @brief Wrapper for HDF5 group
 *
 * @tparam OpType Operation type (read/write)
 */
template <typename OpType> class Group;

/**
 * @brief Template specialization for write operation
 */
template <> class Group<specfem::IO::write> {
public:
  using OpType = specfem::IO::write; ///< Operation type

  /**
   * @name Constructors
   *
   */

  /**
   * @brief Construct a new HDF5 Group object within an HDF5 file with the given
   * name
   *
   * @param file HDF5 file object to create the group in
   * @param name Name of the group
   */
  Group(std::unique_ptr<H5::H5File> &file, const std::string &name)
      : group(std::make_unique<H5::Group>(file->createGroup(name))){};

  /**
   * @brief Construct a new HDF5 Group object within an HDF5 group with the
   * given name
   *
   * @param group HDF5 group object to create the group in
   * @param name Name of the group
   */
  Group(std::unique_ptr<H5::Group> &group, const std::string &name)
      : group(std::make_unique<H5::Group>(group->createGroup(name))){};
  ///@}

  /**
   * @brief Create a new dataset within the group
   *
   * @tparam ViewType Kokkos view type of the data
   * @param name Name of the dataset
   * @param data Data to be written
   * @return specfem::IO::impl::HDF5::Dataset<ViewType, OpType> Dataset object
   */
  template <typename ViewType>
  specfem::IO::impl::HDF5::Dataset<ViewType, OpType>
  createDataset(const std::string &name, const ViewType data) {
    return specfem::IO::impl::HDF5::Dataset<ViewType, OpType>(group, name,
                                                              data);
  }

  /**
   * @brief Create a new group within the group
   *
   * @param name Name of the group
   * @return specfem::IO::impl::HDF5::Group<OpType> Group object
   */
  specfem::IO::impl::HDF5::Group<OpType> createGroup(const std::string &name) {
    return specfem::IO::impl::HDF5::Group<OpType>(group, name);
  }

  ~Group() { group->close(); }

private:
  std::unique_ptr<H5::Group> group; ///< pointer to HDF5 group object
};

/**
 * @brief Template specialization for read operation
 */
template <> class Group<specfem::IO::read> {
public:
  using OpType = specfem::IO::read; ///< Operation type

  /**
   * @name Constructors
   *
   */
  ///@{

  /**
   * @brief Open an existing HDF5 Group object within an HDF5 file with the
   * given name
   *
   * @param file HDF5 file object where the group is located
   * @param name Name of the group
   */
  Group(std::unique_ptr<H5::H5File> &file, const std::string &name)
      : group(std::make_unique<H5::Group>(file->openGroup(name))){};

  /**
   * @brief Open an existing HDF5 Group object within an HDF5 group with the
   * given name
   *
   * @param group HDF5 group object where the group is located
   * @param name Name of the group
   */
  Group(std::unique_ptr<H5::Group> &group, const std::string &name)
      : group(std::make_unique<H5::Group>(group->openGroup(name))){};
  ///@}

  /**
   * @brief Open an existing dataset within the group
   *
   * @tparam ViewType Kokkos view type of the data
   * @param name Name of the dataset
   * @param data Data to be read
   * @return specfem::IO::impl::HDF5::Dataset<ViewType, OpType> Dataset object
   */
  template <typename ViewType>
  specfem::IO::impl::HDF5::Dataset<ViewType, OpType>
  openDataset(const std::string &name, const ViewType data) {
    return specfem::IO::impl::HDF5::Dataset<ViewType, OpType>(group, name,
                                                              data);
  }

  /**
   * @brief Open an existing group within the group
   *
   * @param name Name of the group
   * @return specfem::IO::impl::HDF5::Group<OpType> Group object
   */
  specfem::IO::impl::HDF5::Group<OpType> openGroup(const std::string &name) {
    return specfem::IO::impl::HDF5::Group<OpType>(group, name);
  }

  ~Group() { group->close(); }

private:
  std::unique_ptr<H5::Group> group; ///< pointer to HDF5 group object
};

#endif

} // namespace HDF5
} // namespace impl
} // namespace IO
} // namespace specfem

#endif
