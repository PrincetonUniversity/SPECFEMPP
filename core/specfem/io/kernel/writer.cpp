#include "specfem/io/kernel/writer.hpp"
#include "specfem/io_backends/ADIOS2/ADIOS2.hpp"
#include "specfem/io_backends/ASCII/ASCII.hpp"
#include "specfem/io_backends/HDF5/HDF5.hpp"
#include "specfem/io_backends/NPY/NPY.hpp"
#include "specfem/io_backends/NPZ/NPZ.hpp"

#include "specfem/io/impl/medium_writer.tpp"
#include "specfem/io/kernel/writer.tpp"

// Explicit instantiation

template class specfem::io::kernel_writer<
    specfem::io_backends::HDF5<specfem::io::write> >;

template class specfem::io::kernel_writer<
    specfem::io_backends::ADIOS2<specfem::io::write> >;

template class specfem::io::kernel_writer<
    specfem::io_backends::ASCII<specfem::io::write> >;

template class specfem::io::kernel_writer<
    specfem::io_backends::NPY<specfem::io::write> >;

template class specfem::io::kernel_writer<
    specfem::io_backends::NPZ<specfem::io::write> >;
