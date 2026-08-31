#include "specfem/linear_system/damping_assembler.hpp"

#ifdef SPECFEM_ENABLE_TRILINOS

#include "specfem/assembly/assembly.hpp"
#include "specfem/compute/impl/compute_stiffness_interaction.hpp"
#include "specfem/linear_system/element_stiffness.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>
#include <array>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <vector>

template <typename Tags>
specfem::linear_system::DampingAssembler<Tags>::DampingAssembler(
    AssemblyType &assembly, const LayoutType &layout)
    : assembly_(assembly), layout_(layout) {

  // The probe shares the stiffness probe's scope (NGLL = 5, matching
  // property tag, no attenuation) but tolerates Stacey boundaries -- they
  // are exactly what it measures.
  specfem::linear_system::validate_stiffness_scope<Tags>(
      assembly_, specfem::linear_system::StiffnessScope::with_stacey);
}

template <typename Tags>
Teuchos::RCP<specfem::linear_system::crs_matrix_type>
specfem::linear_system::DampingAssembler<Tags>::assemble() const {
  constexpr auto forward = specfem::simulation::field_type::forward;
  constexpr auto outer = specfem::element::mpi_tag::outer;
  constexpr auto inner = specfem::element::mpi_tag::inner;
  using BaseTags = specfem::tags::Tags<dimension_tag, forward, medium_tag>;

  auto &field = assembly_.fields.template get_simulation_field<forward>();
  const auto &field_impl = field.template get_field<medium_tag>();
  const auto h_u = field_impl.get_host_field();
  const auto h_v = field_impl.get_host_field_dot();
  const auto h_a = field_impl.get_host_field_dot_dot();
  const int nglob = layout_.nglob();

  // Probed blocks: block(p, r, c) = C_p(r, c). Interior points stay exactly
  // zero -- the u = 0 stiffness path adds exact zeros everywhere and the
  // Stacey traction touches boundary points only -- so a nonzero test is an
  // exact damping-point mask, not a tolerance.
  Kokkos::View<type_real ***, Kokkos::HostSpace> block(
      "specfem::linear_system::damping_blocks", nglob, ncomp, ncomp);

  for (int c = 0; c < ncomp; ++c) {
    Kokkos::deep_copy(h_u, 0);
    Kokkos::deep_copy(h_v, 0);
    Kokkos::deep_copy(h_a, 0);
    for (int iglob = 0; iglob < nglob; ++iglob) {
      h_v(iglob, c) = 1;
    }
    assembly_.fields.copy_to_device();

    // istep 0 is a placeholder: the probe runs in the setup phase, before
    // any time loop touches the boundary-value recording.
    specfem::compute::impl::compute_stiffness_interaction<
        5, specfem::tags::expand<BaseTags, outer>>(assembly_, 0);
    specfem::compute::impl::compute_stiffness_interaction<
        5, specfem::tags::expand<BaseTags, inner>>(assembly_, 0);
    assembly_.fields.copy_to_host();

    for (int iglob = 0; iglob < nglob; ++iglob) {
      for (int r = 0; r < ncomp; ++r) {
        block(iglob, r, c) = -h_a(iglob, r);
      }
    }
  }

  // Leave the probe scratch as found (all fields zero, host and device).
  Kokkos::deep_copy(h_u, 0);
  Kokkos::deep_copy(h_v, 0);
  Kokkos::deep_copy(h_a, 0);
  assembly_.fields.copy_to_device();

  std::vector<bool> is_damping_point(nglob, false);
  for (int iglob = 0; iglob < nglob; ++iglob) {
    for (int r = 0; r < ncomp && !is_damping_point[iglob]; ++r) {
      for (int c = 0; c < ncomp; ++c) {
        if (block(iglob, r, c) != 0) {
          is_damping_point[iglob] = true;
          break;
        }
      }
    }
  }

  // Compact structure from the layout: ncomp entries per row at damping
  // points, empty rows elsewhere -- a K-sized graph would waste a full
  // matrix of memory on a block-diagonal operator. Sharing the layout with
  // the stiffness matrix is what guarantees every entry below also exists in
  // K's graph, which the implicit Newmark operator relies on.
  auto matrix = layout_.block_diagonal_matrix(
      [&is_damping_point](const int iglob) { return is_damping_point[iglob]; });

  // scatter_point_block wants a contiguous (ncomp, ncomp) view, and a
  // subview of `block` at one point is LayoutStride -- copy through a small
  // scratch block instead.
  typename LayoutType::host_field_view_type point_block(
      "specfem::linear_system::damping_point_block", ncomp, ncomp);
  for (int iglob = 0; iglob < nglob; ++iglob) {
    if (!is_damping_point[iglob]) {
      continue;
    }
    for (int r = 0; r < ncomp; ++r) {
      for (int c = 0; c < ncomp; ++c) {
        point_block(r, c) = block(iglob, r, c);
      }
    }
    layout_.scatter_point_block(*matrix, iglob, point_block);
  }
  matrix->fillComplete(layout_.owned_map(), layout_.owned_map());

  return matrix;
}

namespace specfem::linear_system_impl {
/// Tag bundle for the only combination explicitly instantiated for the
/// linear system (issue #1982).
using elastic_isotropic_tags =
    specfem::tags::Tags<specfem::element::dimension_tag::dim3,
                        specfem::element::medium_tag::elastic,
                        specfem::element::property_tag::isotropic,
                        specfem::element::attenuation_tag::none>;
} // namespace specfem::linear_system_impl

// Explicit instantiation: 3D elastic isotropic
template class specfem::linear_system::DampingAssembler<
    specfem::linear_system_impl::elastic_isotropic_tags>;

#endif // SPECFEM_ENABLE_TRILINOS
