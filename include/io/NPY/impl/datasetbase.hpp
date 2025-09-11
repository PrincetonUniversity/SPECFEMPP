#pragma once

#include "io/operators.hpp"
#include "npy_header.hpp"
#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace specfem::io::impl::NPY {
template <typename OpType> class DatasetBase;

template <> class DatasetBase<specfem::io::write> {
protected:
  DatasetBase(const boost::filesystem::path &folder_path,
              const std::string &name, std::vector<size_t> dims)
      : data_path(folder_path / boost::filesystem::path(name + ".npy")),
        dims(dims) {}

  template <typename value_type> void write(const value_type *data) const {
    // Delete the file if it exists
    if (boost::filesystem::exists(data_path)) {
      std::ostringstream oss;
      oss << "WARNING : File " << data_path << " already exists. Deleting it.";
      std::cout << oss.str() << std::endl;
      boost::filesystem::remove(data_path);
    }

    std::ofstream file(data_path.string(), std::ios::out | std::ios::binary);
    if (!file.is_open()) {
      std::ostringstream oss;
      oss << "ERROR : Could not open file " << data_path;
      throw std::runtime_error(oss.str());
    }

    int total_elements = 1;
    for (int i = 0; i < dims.size(); ++i) {
      total_elements *= dims[i];
    }

    NPYString header = create_npy_header<value_type>(dims);
    file.write(&header[0], header.size());
    file.write(reinterpret_cast<const char *>(data),
               total_elements * sizeof(value_type));

    file.close();
  }

  void close() const {};

private:
  boost::filesystem::path data_path;
  const std::vector<size_t> dims;
};

template <> class DatasetBase<specfem::io::read> {
protected:
  DatasetBase(const boost::filesystem::path &folder_path,
              const std::string &name, const std::vector<size_t> &dims)
      : data_path(folder_path / boost::filesystem::path(name + ".npy")),
        dims(dims) {}

  template <typename value_type> void read(value_type *data) const {
    std::ifstream file(data_path.string(), std::ios::in | std::ios::binary);
    if (!file.is_open()) {
      std::ostringstream oss;
      oss << "ERROR : Could not open file " << data_path;
      throw std::runtime_error(oss.str());
    }

    // Count total elements
    int total_elements = 1;
    int rank = dims.size();
    for (int i = 0; i < rank; ++i) {
      total_elements *= dims[i];
    }

    std::vector<size_t> shape = parse_npy_header<value_type>(file);
    if (rank != shape.size()) {
      std::ostringstream oss;
      oss << "ERROR : Rank mismatch between dataset and file";
      throw std::runtime_error(oss.str());
    }

    for (int i = 0; i < rank; ++i) {
      if (dims[i] != shape[i]) {
        std::ostringstream oss;
        oss << "ERROR : Dimension mismatch between dataset and file";
        throw std::runtime_error(oss.str());
      }
    }

    file.read(reinterpret_cast<char *>(data),
              total_elements * sizeof(value_type));

    file.close();
  }

  void close() const {}

private:
  boost::filesystem::path data_path;
  const std::vector<size_t> dims;
};

} // namespace specfem::io::impl::NPY
