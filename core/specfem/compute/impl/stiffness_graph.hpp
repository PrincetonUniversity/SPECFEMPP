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
 * @brief Whether the elements of `Tags` run the TensorOperations stiffness
 * graph rather than the legacy kernels.
 *
 * True when every tag except the MPI tag matches @ref stiffness_graph_tags.
 * Only consulted in builds with `SPECFEM_ENABLE_TENSOROPS`.
 *
 * @tparam Tags Compile-time element tags (dimension, medium, property,
 *              attenuation, boundary, wavefield, mpi)
 */
template <typename Tags>
inline constexpr bool uses_stiffness_graph =
    Tags::dimension_tag == stiffness_graph_tags::dimension_tag &&
    Tags::medium_tag == stiffness_graph_tags::medium_tag &&
    Tags::property_tag == stiffness_graph_tags::property_tag &&
    Tags::attenuation_tag == stiffness_graph_tags::attenuation_tag &&
    Tags::boundary_tag == stiffness_graph_tags::boundary_tag &&
    Tags::wavefield_tag == stiffness_graph_tags::wavefield_tag;

/**
 * @brief Compute the stiffness interaction with a TensorOperations level graph.
 *
 * Adds the internal force of every element in `elements` to the acceleration
 * of the `Tags::wavefield_tag` field, exactly as the legacy gather/scatter
 * kernels do for these tags: gradient of the displacement, stress, then the
 * negated weighted divergence. The quadrature weights
 * \f$ w_i w_j w_k \f$ are folded into the stress integrand, so the
 * divergence contracts with the plain Lagrange derivative matrix.
 *
 * Runs on the TensorOperations Team backend, one element per team. Does not
 * fence.
 *
 * @tparam NGLL Number of GLL points per element edge (only 5 instantiated)
 * @tparam Tags Compile-time tags, without an MPI tag
 * @param assembly The assembly object containing the mesh, fields, jacobian
 *        matrix and properties
 * @param elements Contiguous range of elements of `Tags::medium_tag` to update
 * @throws std::runtime_error If SPECFEM++ is built without
 *         `SPECFEM_ENABLE_TENSOROPS`, or the mesh grid does not match `NGLL`
 * @throws std::out_of_range If `elements` reaches outside the medium's
 *         elements
 */
template <int NGLL, typename Tags>
void compute_stiffness_graph(
    const specfem::assembly::assembly<Tags::dimension_tag> &assembly,
    const specfem::datatype::ElementIndexRange &elements);

} // namespace specfem::compute::impl
