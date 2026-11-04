#pragma once

#include "specfem/io_backends/NPY/impl/npy_header.hpp"

#ifndef NO_HDF5
#include "H5Cpp.h"
#endif
#ifndef NO_ADIOS2
#include <adios2.h>
#endif

#include <cstddef>
#include <fstream>
#include <initializer_list>
#include <stdexcept>
#include <string>
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

template <typename T> RawArray<T> read_npy_raw(const std::string &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    throw std::runtime_error("read_npy_raw: cannot open " + path);
  const auto header =
      specfem::io_backends::impl::NPY::parse_npy_header<T>(stream);
  RawArray<T> out;
  out.shape = header.shape;
  out.fortran_order = header.fortran_order;
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
