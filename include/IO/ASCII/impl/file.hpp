#ifndef _SPECFEM_IO_ASCII_IMPL_FILE_HPP
#define _SPECFEM_IO_ASCII_IMPL_FILE_HPP

#include "group.hpp"
#include <boost/filesystem.hpp>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace specfem {
namespace IO {
namespace impl {
namespace ASCII {

// Forward declaration
template <typename OpType> class Group;
template <typename ViewType, typename OpType> class Dataset;

template <typename OpType> class File;

template <> class File<specfem::IO::write> {
public:
  using OpType = specfem::IO::write;

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

  File(const char *name) : File(std::string(name)) {}

  template <typename ViewType>
  specfem::IO::impl::ASCII::Dataset<ViewType, OpType>
  createDataset(const std::string &name, const ViewType data) {
    return specfem::IO::impl::ASCII::Dataset<ViewType, OpType>(folder_path,
                                                               name, data);
  }

  specfem::IO::impl::ASCII::Group<OpType> createGroup(const std::string &name) {
    return specfem::IO::impl::ASCII::Group<OpType>(folder_path, name);
  }

  ~File() {}

private:
  boost::filesystem::path folder_path;
};

template <> class File<specfem::IO::read> {
public:
  using OpType = specfem::IO::read;

  File(const std::string &name) : folder_path(name) {
    if (!boost::filesystem::exists(folder_path)) {
      throw std::runtime_error("ERROR : Folder " + name + " does not exist.");
    }
  }

  template <typename ViewType>
  specfem::IO::impl::ASCII::Dataset<ViewType, OpType>
  openDataset(const std::string &name, const ViewType data) {
    return specfem::IO::impl::ASCII::Dataset<ViewType, OpType>(folder_path,
                                                               name, data);
  }

  specfem::IO::impl::ASCII::Group<OpType> openGroup(const std::string &name) {
    return specfem::IO::impl::ASCII::Group<OpType>(folder_path, name);
  }

  ~File() {}

private:
  boost::filesystem::path folder_path;
};
} // namespace ASCII
} // namespace impl
} // namespace IO
} // namespace specfem

#endif /* _SPECFEM_IO_ASCII_IMPL_FILE_HPP */
