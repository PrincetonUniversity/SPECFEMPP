#pragma once

#include "io/operators.hpp"
#include "native_type.hpp"
#include "native_type.tpp"
#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace specfem::io::impl::ASCII {
template <typename OpType> class DatasetBase;

template <> class DatasetBase<specfem::io::write> {
protected:
  DatasetBase(const boost::filesystem::path &folder_path,
              const std::string &name, const int rank, const int *dims)
      : data_path(folder_path / boost::filesystem::path(name + ".bin")),
        meta_path(folder_path / boost::filesystem::path(name + ".meta")),
        rank(rank), dims(dims) {}

  template <typename value_type> void write(const value_type *data) const {
    // Delete the file if it exists
    if (boost::filesystem::exists(data_path)) {
      std::ostringstream oss;
      oss << "WARNING : File " << data_path << " already exists. Deleting it.";
      std::cout << oss.str() << std::endl;
      boost::filesystem::remove(data_path);
      boost::filesystem::remove(meta_path);
    }

    std::ofstream metadata(meta_path.string());
    if (!metadata.is_open()) {
      std::ostringstream oss;
      oss << "ERROR : Could not open file " << meta_path;
      throw std::runtime_error(oss.str());
    }

    metadata << "type " << native_type<value_type>::string() << "\n";
    metadata << "rank " << rank << "\n";
    metadata << "dims ";
    for (int i = 0; i < rank; ++i) {
      metadata << dims[i] << " ";
    }
    metadata << "\n";

    metadata.close();

    std::ofstream file(data_path.string(), std::ios::out | std::ios::binary);
    if (!file.is_open()) {
      std::ostringstream oss;
      oss << "ERROR : Could not open file " << data_path;
      throw std::runtime_error(oss.str());
    }

    int total_elements = 1;
    for (int i = 0; i < rank; ++i) {
      total_elements *= dims[i];
    }

    file.write(reinterpret_cast<const char *>(data),
               total_elements * sizeof(value_type));

    file.close();
  }

  void close() const {};

private:
  boost::filesystem::path data_path;
  boost::filesystem::path meta_path;
  const int rank;
  const int *dims;
};

template <> class DatasetBase<specfem::io::read> {
protected:
  DatasetBase(const boost::filesystem::path &folder_path,
              const std::string &name, const int rank, const int *dims)
      : data_path(folder_path / boost::filesystem::path(name + ".bin")),
        meta_path(folder_path / boost::filesystem::path(name + ".meta")),
        rank(rank), dims(dims) {}

  template <typename value_type> void read(value_type *data) const {
    // Read meta data file and check if the dimensions match
    if (!boost::filesystem::exists(meta_path)) {
      std::ostringstream oss;
      oss << "ERROR : Metadata file " << meta_path << " does not exist";
      throw std::runtime_error(oss.str());
    }
    std::ifstream metadata(meta_path.string());
    if (!metadata.is_open()) {
      std::ostringstream oss;
      oss << "ERROR : Could not open file " << meta_path;
      throw std::runtime_error(oss.str());
    }

    std::string line;
    std::string token;

    std::getline(metadata, line);
    std::istringstream iss(line);
    iss >> token;
    if (token != "type") {
      std::ostringstream oss;
      oss << "ERROR : Metadata file " << meta_path << " is corrupted";
      throw std::runtime_error(oss.str());
    }
    iss >> token;
    if (token != native_type<value_type>::string()) {
      std::ostringstream oss;
      oss << "ERROR : Metadata file " << meta_path << " type does not match";
      oss << "Expected type: " << native_type<value_type>::string()
          << ", but got: " << token;
      throw std::runtime_error(oss.str());
    }

    std::getline(metadata, line);
    iss = std::istringstream(line);
    iss >> token;
    if (token != "rank") {
      std::ostringstream oss;
      oss << "ERROR : Metadata file " << meta_path << " is corrupted";
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
      oss << "ERROR : Metadata file " << meta_path << " is corrupted";
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

    std::ifstream file(data_path.string(), std::ios::in | std::ios::binary);
    if (!file.is_open()) {
      std::ostringstream oss;
      oss << "ERROR : Could not open file " << data_path;
      throw std::runtime_error(oss.str());
    }

    // Count total elements
    int total_elements = 1;
    for (int i = 0; i < rank; ++i) {
      total_elements *= dims[i];
    }

    file.read(reinterpret_cast<char *>(data),
              total_elements * sizeof(value_type));

    file.close();
  }

  void close() const {}

private:
  boost::filesystem::path data_path;
  boost::filesystem::path meta_path;
  const int rank;
  const int *dims;
};

} // namespace specfem::io::impl::ASCII
