#include "specfem/linear_system/damping_assembler.hpp"

#ifdef SPECFEM_ENABLE_TRILINOS

#include "specfem/assembly/assembly.hpp"
#include "specfem/compute/impl/compute_stiffness_interaction.hpp"
#include "specfem/linear_system/element_stiffness.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>
#include <optional>
#include <sstream>
#include <stdexcept>

template <typename Tags>
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3)
specfem::linear_system::DampingAssembler<Tags>::DampingAssembler(
    AssemblyType &assembly, const DofMap &dof_map)
    : assembly_(assembly), dof_map_(dof_map) {
  validate();
}

template <typename Tags>
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3)
specfem::linear_system::DampingAssembler<Tags>::DampingAssembler(
    AssemblyType &assembly, const FEAssemblyType &fe)
    : assembly_(assembly), dof_map_(assembly, Tags{}), fe_(&fe) {
  validate();
}

template <typename Tags>
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3)
void specfem::linear_system::DampingAssembler<Tags>::validate() const {

  // The probe shares the stiffness probe's scope (NGLL = 5, matching
  // property tag, no attenuation) but tolerates Stacey boundaries -- they
  // are exactly what it measures.
  specfem::linear_system::validate_stiffness_scope<Tags>(
      assembly_, specfem::linear_system::StiffnessScope::with_stacey);

  const auto &field = assembly_.fields.template get_simulation_field<
      specfem::simulation::field_type::forward>();
  const auto &field_impl = field.template get_field<medium_tag>();
  if (field_impl.nglob != dof_map_.nglob()) {
    throw std::runtime_error(
        "specfem::linear_system::DampingAssembler: the dof map does not "
        "match the assembly's forward field.");
  }
}

template <typename Tags>
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3)
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
  // Built up front so that the dof numbering, the sparsity graph and the
  // absorbing-boundary mask all come from one description of the mesh. Built
  // here only when the caller did not supply one to share.
  const FEAssemblyType *fe = fe_;
  const std::optional<FEAssemblyType> owned_fe =
      fe_ ? std::nullopt
          : std::optional<FEAssemblyType>(MappingType(assembly_));
  if (fe == nullptr) {
    fe = &owned_fe.value();
  }

  const auto &mapping = fe->mapping();
  const int nglob = mapping.nglob();

  // Probed blocks: block(p, r, c) = C_p(r, c). Interior points stay exactly
  // zero -- the u = 0 stiffness path adds exact zeros everywhere and the
  // Stacey traction touches boundary points only -- which is what the mask
  // check below relies on, as an exact test rather than a tolerance.
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

  // The sparsity comes from the mesh's absorbing-boundary tags, not from the
  // probe's own nonzero pattern. The two must agree: a block the probe
  // produces at an untagged point has no row to land in and would be dropped
  // silently, so check rather than trust.
  int ndamping = 0;
  for (int iglob = 0; iglob < nglob; ++iglob) {
    if (mapping.is_damping_point(iglob)) {
      ++ndamping;
      continue;
    }
    for (int r = 0; r < ncomp; ++r) {
      for (int c = 0; c < ncomp; ++c) {
        if (block(iglob, r, c) != 0) {
          std::ostringstream message;
          message << "specfem::linear_system::DampingAssembler: the velocity "
                     "probe produced a nonzero damping block at point "
                  << iglob
                  << ", which the mesh does not tag as an absorbing boundary. "
                     "The boundary tags and the Stacey traction disagree.";
          throw std::runtime_error(message.str());
        }
      }
    }
  }

  // Compact to the damping points, so that the update's leading index runs
  // over the same set the dof selection names. C is block-diagonal, so this is
  // one ncomp x ncomp block per boundary point and interior points contribute
  // no rows at all.
  Kokkos::View<int *, Kokkos::HostSpace> damping_points(
      "specfem::linear_system::damping_points", ndamping);
  Kokkos::View<type_real ***, Kokkos::LayoutRight, Kokkos::HostSpace> blocks(
      "specfem::linear_system::damping_point_blocks", ndamping, ncomp, ncomp);

  for (int iglob = 0, p = 0; iglob < nglob; ++iglob) {
    if (!mapping.is_damping_point(iglob)) {
      continue;
    }
    damping_points(p) = iglob;
    for (int r = 0; r < ncomp; ++r) {
      for (int c = 0; c < ncomp; ++c) {
        blocks(p, r, c) = block(iglob, r, c);
      }
    }
    ++p;
  }

  SparseMatrixView<MappingType> matrix(fe->damping_matrix_graph(), mapping);
  matrix.begin_fill();
  // One update for the whole operator: the dof set names every damping point's
  // components, and the rank-3 block carries that point's ncomp x ncomp block.
  const auto dofs = mapping(damping_points, Kokkos::ALL);
  matrix(dofs, dofs) += blocks;
  matrix.finalize();

  return matrix.matrix();
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
