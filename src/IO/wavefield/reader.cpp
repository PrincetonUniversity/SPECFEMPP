#include "IO/wavefield/reader.hpp"
#include "IO/ASCII/ASCII.hpp"
#include "IO/HDF5/HDF5.hpp"
#include "IO/reader.hpp"
#include "IO/wavefield/reader.tpp"

// Explicit instantiation
template class specfem::io::wavefield_reader<
    specfem::io::HDF5<specfem::io::read> >;

template class specfem::io::wavefield_reader<
    specfem::io::ASCII<specfem::io::read> >;
