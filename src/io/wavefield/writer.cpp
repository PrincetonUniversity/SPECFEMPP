#include "io/wavefield/writer.hpp"
#include "io/ADIOS2/ADIOS2.hpp"
#include "io/ASCII/ASCII.hpp"
#include "io/HDF5/HDF5.hpp"
#include "io/Npy/Npy.hpp"
#include "io/wavefield/writer.tpp"

// Explicit instantiation
template class specfem::io::wavefield_writer<
    specfem::io::HDF5<specfem::io::write> >;

template class specfem::io::wavefield_writer<
    specfem::io::ASCII<specfem::io::write> >;

template class specfem::io::wavefield_writer<
    specfem::io::Npy<specfem::io::write> >;

template class specfem::io::wavefield_writer<
    specfem::io::ADIOS2<specfem::io::write> >;
