#include "specfem/io/wavefield/reader.hpp"
#include "specfem/io/ADIOS2/ADIOS2.hpp"
#include "specfem/io/ASCII/ASCII.hpp"
#include "specfem/io/HDF5/HDF5.hpp"
#include "specfem/io/NPY/NPY.hpp"
#include "specfem/io/NPZ/NPZ.hpp"
#include "specfem/io/wavefield/reader.tpp"

// Explicit instantiation
template class specfem::io::wavefield_reader<
    specfem::io::HDF5<specfem::io::read> >;

template class specfem::io::wavefield_reader<
    specfem::io::ASCII<specfem::io::read> >;

template class specfem::io::wavefield_reader<
    specfem::io::ADIOS2<specfem::io::read> >;

template class specfem::io::wavefield_reader<
    specfem::io::NPY<specfem::io::read> >;

template class specfem::io::wavefield_reader<
    specfem::io::NPZ<specfem::io::read> >;
