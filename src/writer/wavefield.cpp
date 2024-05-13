#include "writer/wavefield.hpp"
#include "IO/ASCII/ASCII.hpp"
#include "IO/HDF5/HDF5.hpp"
#include "writer/wavefield.tpp"

// Explicit instantiation

template class specfem::writer::wavefield<
    specfem::IO::HDF5<specfem::IO::write> >;

template class specfem::writer::wavefield<
    specfem::IO::ASCII<specfem::IO::write> >;
