#pragma once

#include "algorithms/locate_point.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/source.hpp"

namespace specfem::assembly::sources_impl {

/**
 * @brief Locate sources in the mesh.
 *
 * @tparam DimensionTag Dimension of the mesh
 * @param element_types Type of elements inside the mesh
 * @param mesh mesh struct containing global coordinates, mapping, control nodes
 *             etc.
 * @param sources vector of source objects to be located and given an associated
 *                `medium_tag`
 */
template <specfem::dimension::type DimensionTag>
void locate_sources(
    const specfem::assembly::element_types<DimensionTag> &element_types,
    const specfem::assembly::mesh<DimensionTag> &mesh,
    std::vector<std::shared_ptr<specfem::sources::source<DimensionTag> > >
        &sources);

} // namespace specfem::assembly::sources_impl

#include "locate_sources.tpp"
