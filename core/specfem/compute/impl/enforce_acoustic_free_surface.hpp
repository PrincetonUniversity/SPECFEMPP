#pragma once

#include "specfem/assembly/assembly.hpp"
#include "specfem/element.hpp"
#include "specfem/enums.hpp"
#include "specfem/macros.hpp"
#include "specfem/point.hpp"
#include "specfem/tag_dispatch.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::compute::impl {

// Zero chi, dot_chi, and ddot_chi at all GLL points tagged as
// acoustic_free_surface in one element type (medium + property + boundary).
//
// This mirrors Fortran SPECFEM3D's acoustic_enforce_free_surface() and is
// needed because the stiffness BC only zeroes the stiffness contribution to
// ddot_chi in-kernel; the source kernel has already atomically added its
// contribution to the global acceleration array and that addition is not undone
// by the stiffness BC. Without explicitly zeroing all three fields here, chi
// and dot_chi accumulate at free-surface nodes over time.
template <typename Tags>
void enforce_acoustic_free_surface_core(
    specfem::assembly::assembly<Tags::dimension_tag> &assembly) {

  constexpr auto medium_tag = Tags::medium_tag;
  constexpr auto property_tag = Tags::property_tag;
  constexpr auto boundary_tag = Tags::boundary_tag;
  constexpr auto wavefield = Tags::wavefield_tag;
  constexpr auto dimension_tag = Tags::dimension_tag;
  constexpr auto dim3 = specfem::element::dimension_tag::dim3;
  constexpr auto acoustic_fs =
      specfem::element::boundary_tag::acoustic_free_surface;

  static_assert(
      boundary_tag == specfem::element::boundary_tag::acoustic_free_surface ||
          boundary_tag ==
              specfem::element::boundary_tag::composite_stacey_dirichlet,
      "enforce_acoustic_free_surface_core requires acoustic_free_surface or "
      "composite_stacey_dirichlet boundary tag");

  const auto element_indices = assembly.element_types.get_elements_on_device(
      medium_tag, property_tag, boundary_tag);

  const int nelements = element_indices.size();
  if (nelements == 0)
    return;

  constexpr bool using_simd = false;
  using PointTags = specfem::tags::Tags<dimension_tag, medium_tag, using_simd>;
  using PointDisplacementType = specfem::point::displacement<PointTags>;
  using PointVelocityType = specfem::point::velocity<PointTags>;
  using PointAccelerationType = specfem::point::acceleration<PointTags>;

  auto &field = assembly.fields.template get_simulation_field<wavefield>();

  // Capture Kokkos Views by value (safe for device lambdas on GPU).
  const auto acoustic_fs_index_mapping =
      assembly.boundaries.acoustic_free_surface_index_mapping;
  const auto qp_boundary_tag =
      assembly.boundaries.acoustic_free_surface.quadrature_point_boundary_tag;

  const int ngllz = assembly.mesh.element_grid.ngllz;
  const int ngllx = assembly.mesh.element_grid.ngllx;
  int nglly = 1;
  if constexpr (dimension_tag == dim3) {
    nglly = assembly.mesh.element_grid.nglly;
  }
  const int ngll_per_elem = ngllz * nglly * ngllx;

  Kokkos::parallel_for(
      "specfem::compute::enforce_acoustic_free_surface",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, nelements *
                                                                ngll_per_elem),
      KOKKOS_LAMBDA(const int idx) {
        const int ispec_local = idx / ngll_per_elem;
        const int gll_idx = idx % ngll_per_elem;

        int iz, iy, ix;
        if constexpr (dimension_tag == dim3) {
          iz = gll_idx / (nglly * ngllx);
          iy = (gll_idx / ngllx) % nglly;
          ix = gll_idx % ngllx;
        } else {
          iz = gll_idx / ngllx;
          iy = 0;
          ix = gll_idx % ngllx;
        }

        const int ispec = element_indices(ispec_local);
        const int local_iac = acoustic_fs_index_mapping(ispec);

        // Only zero GLL points that lie on the acoustic free surface face.
        bool is_free_surface;
        if constexpr (dimension_tag == dim3) {
          is_free_surface =
              (qp_boundary_tag(local_iac, iz, iy, ix) == acoustic_fs);
        } else {
          is_free_surface = (qp_boundary_tag(local_iac, iz, ix) == acoustic_fs);
        }
        if (!is_free_surface)
          return;

        int iglob;
        if constexpr (dimension_tag == dim3) {
          iglob = field.template get_iglob<true, medium_tag>(ispec, iz, iy, ix);
        } else {
          iglob = field.template get_iglob<true, medium_tag>(ispec, iz, ix);
        }

        PointDisplacementType disp;
        PointVelocityType vel;
        PointAccelerationType acc;

        for (int c = 0; c < PointDisplacementType::components; ++c) {
          disp(c) = static_cast<type_real>(0);
          vel(c) = static_cast<type_real>(0);
          acc(c) = static_cast<type_real>(0);
        }

        specfem::point::assembly_index<false> asm_index(iglob);
        specfem::assembly::store_on_device(asm_index, field, disp, vel, acc);
      });
}

// Enforce acoustic free surface BCs (zero chi, dot_chi, ddot_chi) for all
// acoustic elements that carry the free surface condition, dispatching over all
// valid property and boundary tag combinations.
template <int NGLL, typename Tags>
void enforce_acoustic_free_surface(
    specfem::assembly::assembly<Tags::dimension_tag> &assembly) {

  specfem::tag_dispatch::for_each(
      specfem::tag_dispatch::dimension_set<Tags::dimension_tag>{} *
          specfem::tag_dispatch::medium_set<Tags::medium_tag>{} *
          PROPERTY_SET(isotropic, anisotropic, isotropic_cosserat) *
          BOUNDARY_SET(acoustic_free_surface, composite_stacey_dirichlet),
      [&]<typename ElementTags>() {
        enforce_acoustic_free_surface_core<
            specfem::tags::expand<ElementTags, Tags::wavefield_tag>>(assembly);
      });
}

} // namespace specfem::compute::impl
