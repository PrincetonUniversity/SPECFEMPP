#pragma once

// Declared unconditionally so callers need no preprocessor branches.
// TensorOperations headers stay in the .cpp sibling, keeping their compile cost
// out of every file that reaches this header.

#include "specfem/datatype/element_index_range.hpp"
#include "specfem/element.hpp"
#include "specfem/enums.hpp"
#include "specfem/simulation.hpp"
#include "specfem/tags.hpp"

namespace specfem::assembly {
template <specfem::element::dimension_tag DimensionTag> struct assembly;
} // namespace specfem::assembly

namespace specfem::compute::impl {

/**
 * @brief Tags of the elements the TensorOperations stiffness graph supports.
 *
 * Carries no MPI tag: inner and outer elements run the same kernel on different
 * element ranges, so they share one instantiation.
 */
using stiffness_graph_tags =
    specfem::tags::Tags<specfem::element::dimension_tag::dim3,
                        specfem::element::medium_tag::elastic,
                        specfem::element::property_tag::isotropic,
                        specfem::element::attenuation_tag::none,
                        specfem::element::boundary_tag::none,
                        specfem::simulation::field_type::forward>;

/**
 * @brief Compute the stiffness interaction with a TensorOperations level graph.
 *
 * @tparam NGLL Number of GLL points per element edge
 * @tparam Tags Compile-time tags, without an MPI tag
 * @param assembly The assembly object containing the mesh
 * @param elements Contiguous range of elements to update
 * @throws std::runtime_error Always, until the graph is implemented
 */
template <int NGLL, typename Tags>
void compute_stiffness_graph(
    const specfem::assembly::assembly<Tags::dimension_tag> &assembly,
    const specfem::datatype::ElementIndexRange &elements);

} // namespace specfem::compute::impl
