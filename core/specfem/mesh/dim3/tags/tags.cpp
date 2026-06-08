/**
 * @file tags.cpp
 * @brief Implementation of MESHFEM3D element tagging system
 *
 * Implements parallel extraction of material classification from MESHFEM3D
 * Materials containers. Boundary tags are not yet derived from mesh boundaries
 * because mesh.boundaries stores all domain faces (not only absorbing ones).
 * When proper absorbing-boundary configuration is available, this can be
 * updated to classify faces as stacey vs. none.
 */
#include "specfem/enums.hpp"
#include "specfem/mesh.hpp"

specfem::mesh::tags<specfem::element::dimension_tag::dim3>::tags(
    const int nspec, specfem::mesh::materials<dimension_tag> &materials) {

  this->nspec = nspec;

  this->tags_container =
      Kokkos::View<specfem::mesh::impl::tags_container *, Kokkos::HostSpace>(
          "specfem::mesh::tags::tags", this->nspec);

  Kokkos::parallel_for(
      "specfem::mesh::tags::copy_tags",
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, nspec),
      [=, *this](const int ispec) {
        const auto [material_tag, property_tag, attenuation_tag] =
            materials.get_material_type(ispec);

        // Boundary tags are set to none: mesh.boundaries stores all domain
        // faces and cannot distinguish absorbing from free-surface faces
        // without simulation configuration information.
        const auto boundary_tag = specfem::element::boundary_tag::none;

        this->tags_container(ispec) = { material_tag, property_tag,
                                        attenuation_tag, boundary_tag };
      });

  return;
}
