#include "specfem/io/wavefield/reader.hpp"
#include "specfem/io_backends/ADIOS2/ADIOS2.hpp"
#include "specfem/io_backends/ASCII/ASCII.hpp"
#include "specfem/io_backends/HDF5/HDF5.hpp"
#include "specfem/io_backends/NPY/NPY.hpp"
#include "specfem/io_backends/NPZ/NPZ.hpp"

// Explicit instantiation
template class specfem::io::wavefield_reader<
    specfem::io_backends::HDF5<specfem::io::read>>;

template class specfem::io::wavefield_reader<
    specfem::io_backends::ASCII<specfem::io::read>>;

template class specfem::io::wavefield_reader<
    specfem::io_backends::ADIOS2<specfem::io::read>>;

template class specfem::io::wavefield_reader<
    specfem::io_backends::NPY<specfem::io::read>>;

template class specfem::io::wavefield_reader<
    specfem::io_backends::NPZ<specfem::io::read>>;
