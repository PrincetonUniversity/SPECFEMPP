#ifndef _SPECFEM_IO_ASCII_IMPL_GROUP_HPP
#define _SPECFEM_IO_ASCII_IMPL_GROUP_HPP

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
template <typename ViewType, typename OpType> class Dataset;
template <typename OpType> class File;

template <typename OpType> class Group;

template <> class Group<specfem::IO::write> {
public:
  using OpType = specfem::IO::write;

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

  Group(boost::filesystem::path parent_directory, const char *name)
      : Group(parent_directory, std::string(name)) {}

  template <typename ViewType>
  specfem::IO::impl::ASCII::Dataset<ViewType, OpType>
  createDataset(const std::string &name, const ViewType data) {
    return specfem::IO::impl::ASCII::Dataset<ViewType, OpType>(folder_path,
                                                               name, data);
  }

  ~Group() {}

private:
  boost::filesystem::path folder_path;
};

template <> class Group<specfem::IO::read> {
public:
  using OpType = specfem::IO::read;

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

  Group(boost::filesystem::path parent_directory, const char *name)
      : Group(parent_directory, std::string(name)) {}

  template <typename ViewType>
  specfem::IO::impl::ASCII::Dataset<ViewType, OpType>
  openDataset(const std::string &name, const ViewType data) {
    return specfem::IO::impl::ASCII::Dataset<ViewType, OpType>(folder_path,
                                                               name, data);
  }

  ~Group() {}

private:
  boost::filesystem::path folder_path;
};

} // namespace ASCII
} // namespace impl
} // namespace IO
} // namespace specfem

#endif /* _SPECFEM_IO_ASCII_IMPL_GROUP_HPP */
