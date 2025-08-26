#ifndef _SPECFEM_IO_ASCII_IMPL_DATASETBASE_HPP
#define _SPECFEM_IO_ASCII_IMPL_DATASETBASE_HPP

#include "io/operators.hpp"
#include "native_type.hpp"
#include "native_type.tpp"
#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace specfem {
namespace io {
namespace impl {
namespace ASCII {
template <typename OpType> class DatasetBase;

template <> class DatasetBase<specfem::io::write> {
protected:
  DatasetBase(const boost::filesystem::path &folder_path,
              const std::string &name, const int rank,
              const std::vector<int> &dims)
      : file_path(folder_path / boost::filesystem::path(name + ".txt")),
        rank(rank), dims(dims) {

    boost::filesystem::path metadata_path =
        folder_path / boost::filesystem::path(name + ".meta");
    // Delete the file if it exists
    if (boost::filesystem::exists(file_path)) {
      std::ostringstream oss;
      oss << "WARNING : File " << name << " already exists. Deleting it.";
      std::cout << oss.str() << std::endl;
      boost::filesystem::remove(file_path);
      boost::filesystem::remove(metadata_path);
    }

    std::ofstream metadata(metadata_path.string());
    if (!metadata.is_open()) {
      std::ostringstream oss;
      oss << "ERROR : Could not open file " << metadata_path;
      throw std::runtime_error(oss.str());
    }

    metadata << "rank " << rank << "\n";
    metadata << "dims ";
    for (int i = 0; i < rank; ++i) {
      metadata << dims[i] << " ";
    }
    metadata << "\n";

    metadata.close();
  }

  template <typename value_type> void write(const value_type *data) const {

    std::ofstream file(file_path.string());
    if (!file.is_open()) {
      std::ostringstream oss;
      oss << "ERROR : Could not open file " << file_path;
      throw std::runtime_error(oss.str());
    }

    // Count total elements
    int total_elements = 1;
    for (int i = 0; i < rank; ++i) {
      total_elements *= dims[i];
    }

    for (int i = 0; i < total_elements; ++i) {
      specfem::io::impl::ASCII::native_type<value_type>::write(file, data[i]);
    }

    file.close();
  }

  void close() const {};

private:
  boost::filesystem::path file_path;
  const int rank;
  const std::vector<int> dims;
};

template <> class DatasetBase<specfem::io::read> {
protected:
  DatasetBase(const boost::filesystem::path &folder_path,
              const std::string &name, const int rank,
              const std::vector<int> &dims)
      : file_path(folder_path / boost::filesystem::path(name + ".txt")),
        rank(rank), dims(dims) {
    // Read meta data file and check if the dimensions match
    boost::filesystem::path metadata_path =
        folder_path / boost::filesystem::path(name + ".meta");
    if (!boost::filesystem::exists(metadata_path)) {
      std::ostringstream oss;
      oss << "ERROR : Metadata file " << metadata_path << " does not exist";
      throw std::runtime_error(oss.str());
    }
    std::ifstream metadata(metadata_path.string());
    if (!metadata.is_open()) {
      std::ostringstream oss;
      oss << "ERROR : Could not open file " << metadata_path;
      throw std::runtime_error(oss.str());
    }

    std::string line;
    std::getline(metadata, line);
    std::istringstream iss(line);
    std::string token;
    iss >> token;
    if (token != "rank") {
      std::ostringstream oss;
      oss << "ERROR : Metadata file " << metadata_path << " is corrupted";
      throw std::runtime_error(oss.str());
    }
    iss >> token;
    int read_rank = std::stoi(token);
    if (read_rank != rank) {
      std::ostringstream oss;
      oss << "Dimension of the dataset do not match the view";
      oss << "Expected rank: " << rank << ", but got: " << read_rank;
      throw std::runtime_error(oss.str());
    }
    std::getline(metadata, line);
    iss = std::istringstream(line);
    iss >> token;
    if (token != "dims") {
      std::ostringstream oss;
      oss << "ERROR : Metadata file " << metadata_path << " is corrupted";
      throw std::runtime_error(oss.str());
    }
    int read_dims;
    for (int i = 0; i < rank; ++i) {
      iss >> token;
      read_dims = std::stoi(token);
      if (read_dims != dims[i]) {
        std::ostringstream oss;
        oss << "Dimension of the dataset do not match the view";
        oss << "Expected dims[" << i << "]: " << dims[i]
            << ", but got: " << read_dims;
        throw std::runtime_error(oss.str());
      }
    }
    metadata.close();
  }

  template <typename value_type> void read(value_type *data) const {
    std::ifstream file(file_path.string());
    if (!file.is_open()) {
      std::ostringstream oss;
      oss << "ERROR : Could not open file " << file_path;
      throw std::runtime_error(oss.str());
    }

    // Count total elements
    int total_elements = 1;
    for (int i = 0; i < rank; ++i) {
      total_elements *= dims[i];
    }

    for (int i = 0; i < total_elements; ++i) {
      specfem::io::impl::ASCII::native_type<value_type>::read(file, data[i]);
    }

    file.close();
  }

  void close() const {};

private:
  boost::filesystem::path file_path;
  const int rank;
  const std::vector<int> dims;
};

} // namespace ASCII
} // namespace impl
} // namespace io
} // namespace specfem

#endif /* _SPECFEM_IO_ASCII_IMPL_DATASETBASE_HPP */
