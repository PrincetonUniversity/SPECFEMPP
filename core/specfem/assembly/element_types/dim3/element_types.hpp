#pragma once

#include "specfem/assembly/element_types/impl.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/macros.hpp"
#include "specfem/mesh.hpp"

namespace specfem::assembly {

namespace element_types_impl {
using element_types_dim3_base = element_types_base<
    specfem::element::dimension_tag::dim3, MEDIUM_SET(elastic, acoustic),
    PROPERTY_SET(isotropic, anisotropic, isotropic_cosserat),
    BOUNDARY_SET(none), ATTENUATION_SET(none, constant_isotropic)>;
} // namespace element_types_impl

/**
 * @brief 3D spectral element type classification and indexing container
 *
 * Stores medium types (elastic, acoustic), material properties (isotropic,
 * anisotropic, Cosserat), and boundary conditions for each 3D spectral
 * element, with both host and device views for hybrid CPU-GPU computations.
 *
 * @code
 * specfem::assembly::element_types<specfem::element::dimension_tag::dim3>
 *   etypes(nspec, element_grid, mesh, tags);
 *
 * auto elastic = etypes.get_elements_on_device(
 *     specfem::element::medium_tag::elastic);
 * @endcode
 */
template <>
struct element_types<specfem::element::dimension_tag::dim3>
    : public element_types_impl::element_types_dim3_base {

  using base_type = element_types_impl::element_types_dim3_base;

public:
  using base_type::base_type;
};

} // namespace specfem::assembly
