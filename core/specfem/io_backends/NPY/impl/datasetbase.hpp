#pragma once

#include "npy_header.hpp"
#include "specfem/io/operators.hpp"
#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace specfem::io_backends::impl::NPY {
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

  /**
   * @brief Write strings as a NumPy byte-string (`|S<n>`) array
   *
   * The generic overload would copy the std::string object representation,
   * which holds process-local pointers.
   *
   * @param data Strings to write
   */
  void write(const std::string *data) const {
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

    size_t total_elements = 1;
    for (int i = 0; i < dims.size(); ++i) {
      total_elements *= dims[i];
    }

    const size_t width = string_record_width(data, total_elements);
    const std::vector<char> records =
        pack_string_records(data, total_elements, width);

    NPYString header = create_npy_header(dims, map_type<std::string>(), width,
                                         /*fortran_order=*/true);
    file.write(&header[0], header.size());
    file.write(records.data(), records.size());

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

  /**
   * @brief Read a NumPy byte-string (`|S<n>`) array into strings
   *
   * @param data Output strings
   */
  void read(std::string *data) const {
    std::ifstream file(data_path.string(), std::ios::in | std::ios::binary);
    if (!file.is_open()) {
      std::ostringstream oss;
      oss << "ERROR : Could not open file " << data_path;
      throw std::runtime_error(oss.str());
    }

    size_t total_elements = 1;
    for (int i = 0; i < dims.size(); ++i) {
      total_elements *= dims[i];
    }

    const NPYHeader header = read_npy_header(file, map_type<std::string>());
    if (header.shape != dims) {
      std::ostringstream oss;
      oss << "ERROR : Dimension mismatch between dataset and file";
      throw std::runtime_error(oss.str());
    }

    std::vector<char> records(total_elements * header.item_size);
    file.read(records.data(), records.size());
    if (file.gcount() != static_cast<std::streamsize>(records.size())) {
      std::ostringstream oss;
      oss << "ERROR : Unexpected end of file " << data_path;
      throw std::runtime_error(oss.str());
    }
    unpack_string_records(records, total_elements, header.item_size, data);

    file.close();
  }

  void close() const {}

private:
  boost::filesystem::path data_path;
  const std::vector<size_t> dims;
};

} // namespace specfem::io_backends::impl::NPY
