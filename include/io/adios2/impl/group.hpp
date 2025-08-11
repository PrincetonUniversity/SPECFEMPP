#pragma once

#ifndef NO_ADIOS2
#include <adios2.h>
#endif

#include "dataset.hpp"
#include "io/operators.hpp"
#include <memory>
#include <string>

namespace specfem {
namespace io {
namespace impl {
namespace ADIOS2 {

#ifdef NO_ADIOS2
template <typename OpType> class Group {
public:
  template <typename... Args> Group(Args &&...args) {
    throw std::runtime_error("SPECFEM++ was not compiled with ADIOS2 support");
  }

  template <typename ViewType>
  specfem::io::impl::ADIOS2::Dataset<ViewType, OpType>
  createDataset(const std::string &name, const ViewType data) {
    throw std::runtime_error("SPECFEM++ was not compiled with ADIOS2 support");
  }

  specfem::io::impl::ADIOS2::Group<OpType>
  createGroup(const std::string &name) {
    throw std::runtime_error("SPECFEM++ was not compiled with ADIOS2 support");
  }

  template <typename ViewType>
  specfem::io::impl::ADIOS2::Dataset<ViewType, OpType>
  openDataset(const std::string &name, const ViewType data) {
    throw std::runtime_error("SPECFEM++ was not compiled with ADIOS2 support");
  }

  specfem::io::impl::ADIOS2::Group<OpType> openGroup(const std::string &name) {
    throw std::runtime_error("SPECFEM++ was not compiled with ADIOS2 support");
  }
};

#else

// Forward declaration
template <typename ViewType, typename OpType> class Dataset;
template <typename OpType> class File;

/**
 * @brief Wrapper for ADIOS2 group
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

  /**
   * @brief Construct a new ADIOS2 Group object with the given name
   *
   * @param io ADIOS2 IO object
   * @param engine ADIOS2 engine object
   * @param name Name of the group
   */
  Group(std::shared_ptr<adios2::IO> &io,
        std::shared_ptr<adios2::Engine> &engine, const std::string &name)
      : io_ptr(io), engine_ptr(engine), group_name(name) {};

  /**
   * @brief Construct a new ADIOS2 Group object within another group
   *
   * @param parent_io ADIOS2 IO object
   * @param parent_engine ADIOS2 engine object
   * @param parent_group_name Parent group name
   * @param name Name of the group
   */
  Group(std::shared_ptr<adios2::IO> &parent_io,
        std::shared_ptr<adios2::Engine> &parent_engine,
        const std::string &parent_group_name, const std::string &name)
      : io_ptr(parent_io), engine_ptr(parent_engine),
        group_name(parent_group_name + "/" + name) {};
  ///@}

  /**
   * @brief Create a new dataset within the group
   *
   * @tparam ViewType Kokkos view type of the data
   * @param name Name of the dataset
   * @param data Data to be written
   * @return specfem::io::impl::ADIOS2::Dataset<ViewType, OpType> Dataset object
   */
  template <typename ViewType>
  specfem::io::impl::ADIOS2::Dataset<ViewType, OpType>
  createDataset(const std::string &name, const ViewType data) {
    return specfem::io::impl::ADIOS2::Dataset<ViewType, OpType>(
        io_ptr, engine_ptr, group_name + "/" + name, data);
  }

  /**
   * @brief Create a new group within the group
   *
   * @param name Name of the group
   * @return specfem::io::impl::ADIOS2::Group<OpType> Group object
   */
  specfem::io::impl::ADIOS2::Group<OpType>
  createGroup(const std::string &name) {
    return specfem::io::impl::ADIOS2::Group<OpType>(io_ptr, engine_ptr,
                                                    group_name, name);
  }

  ~Group() {}

private:
  std::shared_ptr<adios2::IO> io_ptr;         ///< ADIOS2 IO object
  std::shared_ptr<adios2::Engine> engine_ptr; ///< ADIOS2 engine object
  std::string group_name;                     ///< Full group name path
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
   * @brief Open an existing ADIOS2 Group object with the given name
   *
   * @param io ADIOS2 IO object
   * @param engine ADIOS2 engine object
   * @param name Name of the group
   */
  Group(std::shared_ptr<adios2::IO> &io,
        std::shared_ptr<adios2::Engine> &engine, const std::string &name)
      : io_ptr(io), engine_ptr(engine), group_name(name) {};

  /**
   * @brief Open an existing ADIOS2 Group object within another group
   *
   * @param parent_io ADIOS2 IO object
   * @param parent_engine ADIOS2 engine object
   * @param parent_group_name Parent group name
   * @param name Name of the group
   */
  Group(std::shared_ptr<adios2::IO> &parent_io,
        std::shared_ptr<adios2::Engine> &parent_engine,
        const std::string &parent_group_name, const std::string &name)
      : io_ptr(parent_io), engine_ptr(parent_engine),
        group_name(parent_group_name + "/" + name) {};
  ///@}

  /**
   * @brief Open an existing dataset within the group
   *
   * @tparam ViewType Kokkos view type of the data
   * @param name Name of the dataset
   * @param data Data to be read
   * @return specfem::io::impl::ADIOS2::Dataset<ViewType, OpType> Dataset object
   */
  template <typename ViewType>
  specfem::io::impl::ADIOS2::Dataset<ViewType, OpType>
  openDataset(const std::string &name, const ViewType data) {
    return specfem::io::impl::ADIOS2::Dataset<ViewType, OpType>(
        io_ptr, engine_ptr, group_name + "/" + name, data);
  }

  /**
   * @brief Open an existing group within the group
   *
   * @param name Name of the group
   * @return specfem::io::impl::ADIOS2::Group<OpType> Group object
   */
  specfem::io::impl::ADIOS2::Group<OpType> openGroup(const std::string &name) {
    return specfem::io::impl::ADIOS2::Group<OpType>(io_ptr, engine_ptr,
                                                    group_name, name);
  }

  ~Group() {}

private:
  std::shared_ptr<adios2::IO> io_ptr;         ///< ADIOS2 IO object
  std::shared_ptr<adios2::Engine> engine_ptr; ///< ADIOS2 engine object
  std::string group_name;                     ///< Full group name path
};

#endif

} // namespace ADIOS2
} // namespace impl
} // namespace io
} // namespace specfem
