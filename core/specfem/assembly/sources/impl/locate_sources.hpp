#pragma once

#include "algorithms/locate_point.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/source.hpp"

namespace specfem::assembly::sources_impl {

template <specfem::dimension::type DimensionTag>
void locate_sources(
    const specfem::assembly::element_types<DimensionTag> &element_types,
    const specfem::assembly::mesh<DimensionTag> &mesh,
    std::vector<std::shared_ptr<specfem::sources::source<DimensionTag> > >
        &sources);

} // namespace specfem::assembly::sources_impl

#include "locate_sources.tpp"
