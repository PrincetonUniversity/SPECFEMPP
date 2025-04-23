#include "io/wavefield/reader.hpp"
#include "io/ASCII/ASCII.hpp"
#include "io/HDF5/HDF5.hpp"
#include "io/wavefield/reader.tpp"

// Explicit instantiation
template class specfem::io::wavefield_reader<
    specfem::io::HDF5<specfem::io::read> >;

template class specfem::io::wavefield_reader<
    specfem::io::ASCII<specfem::io::read> >;
