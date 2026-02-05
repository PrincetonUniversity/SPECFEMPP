#include "compute_seismograms.hpp"
#include "impl/compute_seismograms.hpp"
#include "impl/compute_seismograms.tpp"
#include "specfem/enums.hpp"

// Explicit template instantiation
// 2D, NGLL=5,8, forward and backward wavefields
template void
specfem::compute::compute_seismograms<specfem::simulation::field_type::forward,
                                      specfem::element::dimension_tag::dim2, 5>(
    specfem::assembly::assembly<specfem::element::dimension_tag::dim2> &,
    const int &);

template void
specfem::compute::compute_seismograms<specfem::simulation::field_type::forward,
                                      specfem::element::dimension_tag::dim2, 8>(
    specfem::assembly::assembly<specfem::element::dimension_tag::dim2> &,
    const int &);

template void
specfem::compute::compute_seismograms<specfem::simulation::field_type::backward,
                                      specfem::element::dimension_tag::dim2, 5>(
    specfem::assembly::assembly<specfem::element::dimension_tag::dim2> &,
    const int &);

template void
specfem::compute::compute_seismograms<specfem::simulation::field_type::backward,
                                      specfem::element::dimension_tag::dim2, 8>(
    specfem::assembly::assembly<specfem::element::dimension_tag::dim2> &,
    const int &);

// 3D, NGLL=5,8, forward and backward wavefields
template void
specfem::compute::compute_seismograms<specfem::simulation::field_type::forward,
                                      specfem::element::dimension_tag::dim3, 5>(
    specfem::assembly::assembly<specfem::element::dimension_tag::dim3> &,
    const int &);

template void
specfem::compute::compute_seismograms<specfem::simulation::field_type::forward,
                                      specfem::element::dimension_tag::dim3, 8>(
    specfem::assembly::assembly<specfem::element::dimension_tag::dim3> &,
    const int &);

template void
specfem::compute::compute_seismograms<specfem::simulation::field_type::backward,
                                      specfem::element::dimension_tag::dim3, 5>(
    specfem::assembly::assembly<specfem::element::dimension_tag::dim3> &,
    const int &);

template void
specfem::compute::compute_seismograms<specfem::simulation::field_type::backward,
                                      specfem::element::dimension_tag::dim3, 8>(
    specfem::assembly::assembly<specfem::element::dimension_tag::dim3> &,
    const int &);
