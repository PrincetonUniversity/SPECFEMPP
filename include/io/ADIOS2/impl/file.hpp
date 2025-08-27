#pragma once

#ifndef NO_ADIOS2
#include <adios2.h>
#endif

#include "dataset.hpp"
#include "group.hpp"
#include "io/operators.hpp"
#include <memory>
#include <string>

namespace specfem {
namespace io {
namespace impl {
namespace ADIOS2 {

#ifdef NO_ADIOS2
template <typename OpType> class File {
public:
  File(const std::string &name) {
    throw std::runtime_error("SPECFEM++ was not compiled with ADIOS2 support");
  }
  File(const char *name) {
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

  void flush() {
    throw std::runtime_error("SPECFEM++ was not compiled with ADIOS2 support");
  }
};

#else

// Forward declaration
template <typename OpType> class Group;
template <typename ViewType, typename OpType> class Dataset;

/**
 * @brief Wrapper for ADIOS2 file
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
   * @brief Construct a new ADIOS2 File object with the given name
   *
   * @param name Name of the file
   */
  File(const std::string &name)
      : adios(std::make_shared<adios2::ADIOS>()),
        io(std::make_shared<adios2::IO>(adios->DeclareIO("specfem_io"))),
        engine(std::make_shared<adios2::Engine>(
            io->Open(name + ".bp", adios2::Mode::Write))) {}
  /**
   * @brief Construct a new File object with the given name
   *
   * @param name Name of the file
   */
  File(const char *name)
      : adios(std::make_shared<adios2::ADIOS>()),
        io(std::make_shared<adios2::IO>(adios->DeclareIO("specfem_io"))),
        engine(std::make_shared<adios2::Engine>(
            io->Open(std::string(name) + ".bp", adios2::Mode::Write))) {}
  ///@}

  /**
   * @brief Create a new dataset within the file
   *
   * @tparam ViewType Kokkos view type of the data
   * @param name Name of the dataset
   * @param data Data to be written
   * @return specfem::io::impl::ADIOS2::Dataset<ViewType, OpType> Dataset object
   */
  template <typename ViewType>
  specfem::io::impl::ADIOS2::Dataset<ViewType, OpType>
  createDataset(const std::string &name, const ViewType data) {
    return specfem::io::impl::ADIOS2::Dataset<ViewType, OpType>(io, engine,
                                                                name, data);
  }

  /**
   * @brief Create a new group within the file
   *
   * @param name Name of the group
   * @return specfem::io::impl::ADIOS2::Group<OpType> Group object
   */
  specfem::io::impl::ADIOS2::Group<OpType>
  createGroup(const std::string &name) {
    return specfem::io::impl::ADIOS2::Group<OpType>(io, engine, name);
  }

  void flush() { engine->Flush(); }

  ~File() {
    if (engine) {
      engine->Close();
    }
  }

private:
  std::shared_ptr<adios2::ADIOS> adios;   ///< ADIOS2 context
  std::shared_ptr<adios2::IO> io;         ///< ADIOS2 IO object
  std::shared_ptr<adios2::Engine> engine; ///< ADIOS2 engine for file operations
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
   * @brief Read the ADIOS2 file with the given name
   *
   * @param name Name of the file
   */
  File(const std::string &name)
      : adios(std::make_shared<adios2::ADIOS>()),
        io(std::make_shared<adios2::IO>(adios->DeclareIO("specfem_io"))),
        engine(std::make_shared<adios2::Engine>(
            io->Open(name + ".bp", adios2::Mode::ReadRandomAccess))) {}
  /**
   * @brief Read the ADIOS2 file with the given name
   *
   * @param name Name of the file
   */
  File(const char *name)
      : adios(std::make_shared<adios2::ADIOS>()),
        io(std::make_shared<adios2::IO>(adios->DeclareIO("specfem_io"))),
        engine(std::make_shared<adios2::Engine>(io->Open(
            std::string(name) + ".bp", adios2::Mode::ReadRandomAccess))) {}

  ///@}

  /**
   * @brief Open an existing dataset within the file
   *
   * @tparam ViewType Kokkos view type of the data
   * @param name Name of the dataset
   * @param data Data to be read
   * @return specfem::io::impl::ADIOS2::Dataset<ViewType, OpType> Dataset object
   */
  template <typename ViewType>
  specfem::io::impl::ADIOS2::Dataset<ViewType, OpType>
  openDataset(const std::string &name, const ViewType data) {
    return specfem::io::impl::ADIOS2::Dataset<ViewType, OpType>(io, engine,
                                                                name, data);
  }

  /**
   * @brief Open an existing group within the file
   *
   * @param name Name of the group
   * @return specfem::io::impl::ADIOS2::Group<OpType> Group object
   */
  specfem::io::impl::ADIOS2::Group<OpType> openGroup(const std::string &name) {
    return specfem::io::impl::ADIOS2::Group<OpType>(io, engine, name);
  }

  ~File() {
    if (engine) {
      engine->Close();
    }
  }

private:
  std::shared_ptr<adios2::ADIOS> adios;   ///< ADIOS2 context
  std::shared_ptr<adios2::IO> io;         ///< ADIOS2 IO object
  std::shared_ptr<adios2::Engine> engine; ///< ADIOS2 engine for file operations
};

#endif

} // namespace ADIOS2
} // namespace impl
} // namespace io
} // namespace specfem
