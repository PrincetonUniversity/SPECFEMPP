#pragma once

#include "specfem/assembly/element_types/impl.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/macros.hpp"
#include "specfem/mesh.hpp"

namespace specfem::assembly {

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
    : public element_types_impl::element_types_base<
          specfem::element::dimension_tag::dim3, DIMENSION_SET(dim3),
          MEDIUM_SET(elastic, acoustic),
          PROPERTY_SET(isotropic, anisotropic, isotropic_cosserat),
          BOUNDARY_SET(none), ATTENUATION_SET(none, constant_isotropic)> {

  using base_type = element_types_impl::element_types_base<
      specfem::element::dimension_tag::dim3, DIMENSION_SET(dim3),
      MEDIUM_SET(elastic, acoustic),
      PROPERTY_SET(isotropic, anisotropic, isotropic_cosserat),
      BOUNDARY_SET(none), ATTENUATION_SET(none, constant_isotropic)>;

public:
  element_types() = default;

  /**
   * @brief Construct 3D element types container from mesh and tag data.
   *
   * @param nspec Total number of spectral elements in the mesh
   * @param element_grid GLL grid configuration (ngllz, nglly, ngllx)
   * @param mesh 3D assembly mesh containing geometry and connectivity
   * @param tags Element classification data (medium, property, boundary tags)
   */
  element_types(
      const int nspec,
      const specfem::mesh_entity::element_grid<dimension_tag> &element_grid,
      const specfem::assembly::mesh<dimension_tag> &mesh,
      const specfem::mesh::tags<dimension_tag> &tags)
      : base_type(nspec, element_grid, mesh, tags) {}
};

} // namespace specfem::assembly
