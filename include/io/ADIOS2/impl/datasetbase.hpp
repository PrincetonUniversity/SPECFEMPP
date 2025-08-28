#pragma once

#ifndef NO_ADIOS2
#include <adios2.h>
#endif

#include "io/operators.hpp"
#include "native_type.hpp"
#include <memory>
#include <string>

namespace specfem {
namespace io {
namespace impl {
namespace ADIOS2 {

#ifndef NO_ADIOS2
template <typename OpType> class DatasetBase;

template <> class DatasetBase<specfem::io::write> {
protected:
  DatasetBase(std::shared_ptr<adios2::IO> &io,
              std::shared_ptr<adios2::Engine> &engine, const std::string &name)
      : io_ptr(io), engine_ptr(engine), dataset_name(name) {}

  template <typename value_type>
  void write(const value_type *data,
             const adios2::Variable<value_type> &variable) {
    engine_ptr->Put(variable, data);
  }

  void close() {
    // ADIOS2 handles cleanup automatically with shared_ptr
  }

  ~DatasetBase() { close(); }

protected:
  std::shared_ptr<adios2::IO> io_ptr;
  std::shared_ptr<adios2::Engine> engine_ptr;
  std::string dataset_name;
};

template <> class DatasetBase<specfem::io::read> {
protected:
  DatasetBase(std::shared_ptr<adios2::IO> &io,
              std::shared_ptr<adios2::Engine> &engine, const std::string &name)
      : io_ptr(io), engine_ptr(engine), dataset_name(name) {}

  template <typename value_type>
  void read(value_type *data, adios2::Variable<value_type> &variable) {

    // Get shape of the variable from file
    auto shape = variable.Shape();

    // Prepare selection — full dataset
    std::vector<std::size_t> start(shape.size(), 0);
    std::vector<std::size_t> count = shape;

    // Make a mutable copy of the variable to set selection
    variable.SetSelection({ start, count });

    // Perform read
    engine_ptr->Get(variable, data);
    engine_ptr->PerformGets(); // <--- Required to actually retrieve the data
  }

  void close() {
    // ADIOS2 handles cleanup automatically with shared_ptr
  }

  ~DatasetBase() { close(); }

protected:
  std::shared_ptr<adios2::IO> io_ptr;
  std::shared_ptr<adios2::Engine> engine_ptr;
  std::string dataset_name;
};
#endif
} // namespace ADIOS2
} // namespace impl
} // namespace io
} // namespace specfem
