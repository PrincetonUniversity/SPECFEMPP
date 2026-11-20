#pragma once

#ifndef NO_HDF5
#include "H5Cpp.h"
#endif
#ifndef NO_ADIOS2
#include <adios2.h>
#endif

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

/**
 * @brief Reader-independent oracle for on-disk array layout.
 *
 * Reads a dataset the way an external consumer (numpy, h5py, bpls) would: it
 * trusts only the shape and storage order stored in the file header and
 * computes the offset of a logical index from them.
 */
namespace specfem_test::layout_oracle {

template <typename T> struct RawArray {
  std::vector<std::size_t> shape;
  bool fortran_order = false;
  std::vector<T> data;

  /// Value at logical index @p idx according to shape and storage order
  T at(std::initializer_list<std::size_t> idx) const {
    if (idx.size() != shape.size())
      throw std::out_of_range("RawArray::at: rank mismatch");
    std::size_t offset = 0;
    std::size_t stride = 1;
    if (fortran_order) {
      std::size_t d = 0;
      for (const std::size_t i : idx) {
        if (i >= shape[d])
          throw std::out_of_range("RawArray::at: index out of range");
        offset += i * stride;
        stride *= shape[d];
        ++d;
      }
    } else {
      std::size_t d = shape.size();
      for (auto it = idx.end(); it != idx.begin();) {
        --it;
        --d;
        if (*it >= shape[d])
          throw std::out_of_range("RawArray::at: index out of range");
        offset += *it * stride;
        stride *= shape[d];
      }
    }
    return data.at(offset);
  }
};

inline std::size_t product(const std::vector<std::size_t> &shape) {
  std::size_t n = 1;
  for (const std::size_t s : shape)
    n *= s;
  return n;
}

/// Position just past the `key:` entry of an NPY header dictionary
inline std::size_t npy_value_start(const std::string &header,
                                   const std::string &key) {
  const std::size_t key_start = header.find("'" + key + "'");
  if (key_start == std::string::npos)
    throw std::runtime_error("npy header: no key " + key + " in " + header);
  return header.find_first_not_of(' ', header.find(':', key_start) + 1);
}

/// Shape tuple of an NPY header dictionary
inline std::vector<std::size_t> npy_shape(const std::string &header) {
  const std::size_t open = header.find('(', npy_value_start(header, "shape"));
  const std::size_t close = header.find(')', open);
  if (open == std::string::npos || close == std::string::npos)
    throw std::runtime_error("npy header: malformed shape in " + header);
  std::vector<std::size_t> shape;
  std::size_t pos = open + 1;
  while (pos < close) {
    const char c = header[pos];
    if (c == ',' || c == ' ') {
      ++pos;
      continue;
    }
    std::size_t used = 0;
    shape.push_back(std::stoull(header.substr(pos), &used));
    pos += used;
  }
  return shape;
}

/**
 * @brief Read an NPY file with a parser separate from the specfem NPY backend.
 *
 * Follows the NumPy format description: magic string, version, little-endian
 * header length, then a dictionary with `descr`, `fortran_order` and `shape`.
 *
 * @tparam T Expected element type; a different `descr` throws
 * @param path Path of the `.npy` file
 * @return Shape, storage order and payload from the file
 */
template <typename T> RawArray<T> read_npy_raw(const std::string &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    throw std::runtime_error("read_npy_raw: cannot open " + path);

  char magic[6];
  unsigned char version[2];
  stream.read(magic, sizeof(magic));
  stream.read(reinterpret_cast<char *>(version), sizeof(version));
  if (!stream || std::memcmp(magic, "\x93NUMPY", sizeof(magic)) != 0)
    throw std::runtime_error("read_npy_raw: bad magic string in " + path);

  // Version 1 stores a 2-byte header length; later versions store 4 bytes.
  const int n_length_bytes = version[0] == 1 ? 2 : 4;
  std::uint32_t header_length = 0;
  for (int i = 0; i < n_length_bytes; ++i) {
    unsigned char byte = 0;
    stream.read(reinterpret_cast<char *>(&byte), 1);
    header_length |= static_cast<std::uint32_t>(byte) << (8 * i);
  }
  std::string header(header_length, ' ');
  stream.read(header.data(), header_length);
  if (!stream)
    throw std::runtime_error("read_npy_raw: truncated header in " + path);

  // descr is e.g. '<f8': byte order, kind, element size
  const std::size_t descr_start = npy_value_start(header, "descr") + 1;
  const std::string descr =
      header.substr(descr_start, header.find('\'', descr_start) - descr_start);
  const std::string expected_descr = std::string("<") +
                                     (std::is_floating_point_v<T> ? "f" : "i") +
                                     std::to_string(sizeof(T));
  if (descr != expected_descr)
    throw std::runtime_error("read_npy_raw: descr " + descr + " in " + path +
                             " is not " + expected_descr);

  RawArray<T> out;
  out.shape = npy_shape(header);
  out.fortran_order =
      header.compare(npy_value_start(header, "fortran_order"), 4, "True") == 0;
  out.data.resize(product(out.shape));
  stream.read(reinterpret_cast<char *>(out.data.data()),
              static_cast<std::streamsize>(out.data.size() * sizeof(T)));
  if (!stream)
    throw std::runtime_error("read_npy_raw: truncated payload in " + path);
  return out;
}

#ifndef NO_HDF5
template <typename T>
RawArray<T> read_hdf5_raw(const std::string &file,
                          const std::string &dataset_path) {
  H5::H5File h5(file, H5F_ACC_RDONLY);
  H5::DataSet dataset = h5.openDataSet(dataset_path);
  H5::DataSpace space = dataset.getSpace();
  std::vector<hsize_t> dims(space.getSimpleExtentNdims());
  space.getSimpleExtentDims(dims.data());
  RawArray<T> out;
  out.shape.assign(dims.begin(), dims.end());
  out.fortran_order = false;
  out.data.resize(product(out.shape));
  dataset.read(out.data.data(), dataset.getDataType());
  return out;
}
#endif

#ifndef NO_ADIOS2
template <typename T>
RawArray<T> read_adios2_raw(const std::string &file, const std::string &name) {
  adios2::ADIOS adios;
  adios2::IO io = adios.DeclareIO("oracle");
  adios2::Engine engine = io.Open(file, adios2::Mode::Read);
  engine.BeginStep();
  adios2::Variable<T> var = io.InquireVariable<T>(name);
  if (!var)
    throw std::runtime_error("read_adios2_raw: no variable " + name);
  RawArray<T> out;
  const auto shape = var.Shape();
  out.shape.assign(shape.begin(), shape.end());
  out.fortran_order = false;
  out.data.resize(product(out.shape));
  engine.Get(var, out.data.data(), adios2::Mode::Sync);
  engine.EndStep();
  engine.Close();
  return out;
}
#endif

} // namespace specfem_test::layout_oracle
