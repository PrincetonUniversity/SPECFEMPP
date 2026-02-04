#pragma once

#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/mesh.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly::fields_impl {

template <specfem::dimension::type DimensionTag>
void assign_assembly_index_mapping(
    const specfem::assembly::mesh<DimensionTag> &mesh,
    const specfem::assembly::element_types<DimensionTag> &element_types,
    Kokkos::View<int *, Kokkos::LayoutLeft, Kokkos::HostSpace>
        assembly_index_mapping,
    int &nglob, const specfem::element::medium_tag MediumTag);

} // namespace specfem::assembly::fields_impl
