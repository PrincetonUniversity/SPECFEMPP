#include "IO/wavefield/writer.hpp"
#include "IO/ASCII/ASCII.hpp"
#include "IO/HDF5/HDF5.hpp"
#include "IO/wavefield/writer.tpp"

// Explicit instantiation

template class specfem::IO::wavefield_writer<
    specfem::IO::HDF5<specfem::IO::write> >;

template class specfem::IO::wavefield_writer<
    specfem::IO::ASCII<specfem::IO::write> >;
