#pragma once

#include "algorithms/locate_point.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/source.hpp"

namespace specfem::assembly::sources_impl {

void locate_sources(
    const specfem::assembly::element_types<specfem::dimension::type::dim2>
        &element_types,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    std::vector<std::shared_ptr<specfem::sources::source> > &sources);

} // namespace specfem::assembly::sources_impl
