#pragma once

#include "specfem/assembly/mesh.hpp"
#include "specfem/element.hpp"
#include "specfem/mesh.hpp"
#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace specfem::assembly::element_types_impl {

/**
 * @brief Optional per-element globe (SPECFEM3D_GLOBE) context storage.
 *
 * Groups the globe-only per-element views that a globe mesh carries alongside
 * the medium/property/boundary tag classification. This struct exists only
 * when the assembly was built from a globe mesh; on Cartesian meshes the owning
 * @c element_types holds no instance (see the @c std::optional member there).
 *
 * The struct is parameterised on the view-alias template so it reuses the
 * owning class's @c TagViewType without hardcoding a Kokkos execution space.
 *
 * @tparam TagViewType Alias template mapping an element type @c T to the host
 *                     view type used for per-element storage.
 */
template <template <typename> class TagViewType> struct globe_context {
  /** Radial region of each element. */
  TagViewType<specfem::element::region_tag> regions;
  /** Mesher radial-zone flag (IFLAG_*) of each element. */
  TagViewType<int> idoubling;
  /** Lower radius of each element's radial shell, in metres. Double, not
   *  type_real: setup-only data handed to the double-precision globe model. */
  TagViewType<double> rmin;
  /** Upper radius of each element's radial shell, in metres. */
  TagViewType<double> rmax;
  /** Whether each element is in the crust, as decided by Moho stretching. */
  TagViewType<bool> elem_in_crust;
  /** Whether each element is in the mantle. */
  TagViewType<bool> elem_in_mantle;

  /** @brief Default constructor; leaves all views empty. */
  globe_context() = default;

  /**
   * @brief Allocate and fill the globe context views in compute-domain order.
   *
   * Allocates the six views (size @p nspec) and copies each field from the
   * mesh-domain-ordered @p element_context, translating compute-domain indices
   * to mesh-domain indices via @p mesh.
   *
   * @tparam DimensionTag   Dimension of the simulation (deduced from @p mesh).
   * @param nspec           Number of spectral elements in the compute domain.
   * @param mesh            Compute-to-mesh index mapping.
   * @param element_context Per-element globe context in mesh-domain order.
   * @throws std::runtime_error if @p element_context does not have @p nspec
   *         entries.
   */
  template <specfem::element::dimension_tag DimensionTag>
  globe_context(int nspec, const specfem::assembly::mesh<DimensionTag> &mesh,
                const std::vector<specfem::mesh::globe_element_context>
                    &element_context) {
    if (static_cast<int>(element_context.size()) != nspec) {
      throw std::runtime_error("element_types: globe element context has " +
                               std::to_string(element_context.size()) +
                               " entries for " + std::to_string(nspec) +
                               " elements");
    }
    regions = TagViewType<specfem::element::region_tag>(
        "specfem::assembly::element_types::regions", nspec);
    idoubling =
        TagViewType<int>("specfem::assembly::element_types::idoubling", nspec);
    rmin = TagViewType<double>("specfem::assembly::element_types::rmin", nspec);
    rmax = TagViewType<double>("specfem::assembly::element_types::rmax", nspec);
    elem_in_crust = TagViewType<bool>(
        "specfem::assembly::element_types::elem_in_crust", nspec);
    elem_in_mantle = TagViewType<bool>(
        "specfem::assembly::element_types::elem_in_mantle", nspec);
    for (int ispec = 0; ispec < nspec; ispec++) {
      const auto &context = element_context[mesh.h_compute_to_mesh(ispec)];
      regions(ispec) = context.region;
      idoubling(ispec) = context.idoubling;
      rmin(ispec) = context.rmin;
      rmax(ispec) = context.rmax;
      elem_in_crust(ispec) = context.element_in_crust;
      elem_in_mantle(ispec) = context.element_in_mantle;
    }
  }
};

} // namespace specfem::assembly::element_types_impl
