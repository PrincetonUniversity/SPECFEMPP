#pragma once
#include "kokkos_abstractions.h"
#include "specfem/assembly/mesh.hpp"
#include "specfem/source.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly::compute_source_array_impl {

void from_vector(
    const specfem::sources::vector_source<specfem::dimension::type::dim2>
        &source,
    Kokkos::View<type_real ***, Kokkos::LayoutRight, Kokkos::HostSpace>
        source_array);

} // namespace specfem::assembly::compute_source_array_impl
