#include "writer/wavefield.hpp"
#include "IO/HDF5/HDF5.hpp"
#include "writer/wavefield.tpp"

// Explicit instantiation

template class specfem::writer::wavefield<specfem::IO::HDF5>;
