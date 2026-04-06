#pragma once

#include "specfem/assembly/element_types/impl.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/macros.hpp"
#include "specfem/mesh.hpp"

namespace specfem::assembly {

namespace element_types_impl {
using element_types_dim2_base =
    element_types_base<specfem::element::dimension_tag::dim2,
                       MEDIUM_SET(elastic_psv, elastic_sh, elastic_psv_t,
                                  acoustic, poroelastic, electromagnetic_te),
                       PROPERTY_SET(isotropic, anisotropic, isotropic_cosserat),
                       BOUNDARY_SET(none, stacey, acoustic_free_surface,
                                    composite_stacey_dirichlet),
                       ATTENUATION_SET(none, constant_isotropic)>;
} // namespace element_types_impl

/**
 * @brief 2D spectral element type classification and indexing container
 *
 * Stores medium types (elastic P-SV, elastic SH, acoustic, poroelastic),
 * material properties (isotropic, anisotropic, Cosserat), and boundary
 * conditions for each 2D spectral element, with both host and device views
 * for hybrid CPU-GPU computations.
 *
 * @code
 * specfem::assembly::element_types<specfem::element::dimension_tag::dim2>
 *   etypes(nspec, element_grid, mesh, tags);
 *
 * auto psv = etypes.get_elements_on_device(
 *     specfem::element::medium_tag::elastic_psv);
 * @endcode
 */
template <>
struct element_types<specfem::element::dimension_tag::dim2>
    : public element_types_impl::element_types_dim2_base {

  using base_type = element_types_impl::element_types_dim2_base;

public:
  using base_type::base_type;
};

} // namespace specfem::assembly
