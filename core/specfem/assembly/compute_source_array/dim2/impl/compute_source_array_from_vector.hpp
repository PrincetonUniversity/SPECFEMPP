#pragma once
#include "kokkos_abstractions.h"
#include "specfem/assembly/mesh.hpp"
#include "specfem/source.hpp"

namespace specfem::assembly::compute_source_array_impl {

void from_vector(
    const specfem::sources::vector_source<specfem::dimension::type::dim2>
        &source,
    specfem::kokkos::HostView3d<type_real> source_array);

} // namespace specfem::assembly::compute_source_array_impl
