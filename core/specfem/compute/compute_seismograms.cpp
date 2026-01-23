#include "compute_seismograms.hpp"
#include "enumerations/interface.hpp"
#include "impl/compute_seismograms.hpp"
#include "impl/compute_seismograms.tpp"

// Explicit template instantiation
// 2D, NGLL=5,8, forward and backward wavefields
template void specfem::compute::compute_seismograms<
    specfem::wavefield::simulation_field::forward,
    specfem::dimension::type::dim2, 5>(
    specfem::assembly::assembly<specfem::dimension::type::dim2> &, const int &);

template void specfem::compute::compute_seismograms<
    specfem::wavefield::simulation_field::forward,
    specfem::dimension::type::dim2, 8>(
    specfem::assembly::assembly<specfem::dimension::type::dim2> &, const int &);

template void specfem::compute::compute_seismograms<
    specfem::wavefield::simulation_field::backward,
    specfem::dimension::type::dim2, 5>(
    specfem::assembly::assembly<specfem::dimension::type::dim2> &, const int &);

template void specfem::compute::compute_seismograms<
    specfem::wavefield::simulation_field::backward,
    specfem::dimension::type::dim2, 8>(
    specfem::assembly::assembly<specfem::dimension::type::dim2> &, const int &);

// 3D, NGLL=5,8, forward and backward wavefields
template void specfem::compute::compute_seismograms<
    specfem::wavefield::simulation_field::forward,
    specfem::dimension::type::dim3, 5>(
    specfem::assembly::assembly<specfem::dimension::type::dim3> &, const int &);

template void specfem::compute::compute_seismograms<
    specfem::wavefield::simulation_field::forward,
    specfem::dimension::type::dim3, 8>(
    specfem::assembly::assembly<specfem::dimension::type::dim3> &, const int &);

template void specfem::compute::compute_seismograms<
    specfem::wavefield::simulation_field::backward,
    specfem::dimension::type::dim3, 5>(
    specfem::assembly::assembly<specfem::dimension::type::dim3> &, const int &);

template void specfem::compute::compute_seismograms<
    specfem::wavefield::simulation_field::backward,
    specfem::dimension::type::dim3, 8>(
    specfem::assembly::assembly<specfem::dimension::type::dim3> &, const int &);
