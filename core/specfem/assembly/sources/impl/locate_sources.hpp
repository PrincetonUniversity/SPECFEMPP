#pragma once

#include "specfem/algorithms.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/source.hpp"

namespace specfem::assembly::sources_impl {

/**
 * @brief Locate seismic sources within the finite element mesh
 *
 * Maps source global coordinates to local element coordinates and assigns
 * medium tags based on element classification.
 *
 * @tparam DimensionTag Spatial dimension (`dim2` or `dim3`)
 *
 * @param element_types Element classification data (medium, property, boundary
 * types)
 * @param mesh Finite element mesh with coordinates and connectivity
 * @param sources [in,out] Source objects to locate. Input: coordinates and time
 * functions. Output: assigned element indices and medium tags.
 *
 * @throws std::runtime_error If source cannot be located within mesh domain
 * @throws std::invalid_argument If coordinates are invalid or mesh is malformed
 *
 * @code
 * std::vector<std::shared_ptr<specfem::sources::source<specfem::dimension::type::dim2>>>
 * sources;
 * // ... populate sources
 * locate_sources<specfem::dimension::type::dim2>(element_types, mesh, sources);
 * @endcode
 *
 * @note This function is an implementation detail and should be only called
 * within @ref specfem::assembly::sources construction.
 */
template <specfem::dimension::type DimensionTag>
void locate_sources(
    const specfem::assembly::element_types<DimensionTag> &element_types,
    const specfem::assembly::mesh<DimensionTag> &mesh,
    std::vector<std::shared_ptr<specfem::sources::source<DimensionTag> > >
        &sources);

} // namespace specfem::assembly::sources_impl

#include "locate_sources.tpp"
