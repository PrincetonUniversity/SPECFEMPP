/**
 * @file tags.cpp
 * @brief Implementation of MESHFEM3D element tagging system
 *
 * Implements parallel extraction of material classification from MESHFEM3D
 * Materials containers. Sets boundary tags to `none` since other boundary
 * conditions are not yet implemented.
 */
#include "enumerations/interface.hpp"
#include "specfem/mesh.hpp"

/**
 * @brief Constructor that extracts material tags in parallel
 *
 * Uses Kokkos parallel_for to extract material classification from the
 * Materials container. Sets all boundary tags to `none` since other
 * boundary conditions are not yet implemented.
 *
 * @param nspec Number of spectral elements
 * @param materials Materials container with material data
 */
specfem::mesh::tags<specfem::dimension::type::dim3>::tags(
    const int nspec, specfem::mesh::materials<dimension_tag> &materials) {

  this->nspec = nspec;

  this->tags_container =
      Kokkos::View<specfem::mesh::impl::tags_container *, Kokkos::HostSpace>(
          "specfem::mesh::tags::tags", this->nspec);

  Kokkos::parallel_for(
      "specfem::mesh::tags::copy_tags",
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, nspec),
      [=, *this](const int ispec) {
        const auto [material_tag, property_tag] =
            materials.get_material_type(ispec);

        // Set to none since other boundary conditions are not yet implemented
        const auto boundary_tag = specfem::element::boundary_tag::none;

        this->tags_container(ispec) = { material_tag, property_tag,
                                        boundary_tag };
      });

  return;
}
