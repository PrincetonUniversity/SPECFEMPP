#pragma once

#include "specfem/assembly.hpp"

namespace specfem::compute::impl {

template <int NGLL, typename Tags>
void compute_coupling_conjugate_integral_nonconforming(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly)
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3);

template <int NGLL, typename Tags>
void compute_coupling_conjugate_integral_nonconforming(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim2>
        &assembly)
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim2);

} // namespace specfem::compute::impl
