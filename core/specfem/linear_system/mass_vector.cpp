#include "specfem/linear_system/mass_vector.hpp"

#ifdef SPECFEM_ENABLE_TRILINOS

#include "specfem/assembly/assembly.hpp"
#include "specfem/compute/initialize_mass_matrix.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>
#include <Tpetra_Vector.hpp>
#include <cstddef>
#include <stdexcept>

template <typename Tags>
Teuchos::RCP<specfem::linear_system::vector_type>
specfem::linear_system::assemble_mass_vector(
    specfem::assembly::assembly<Tags::dimension_tag> &assembly,
    const DofMap &dof_map) {

  static_assert(Tags::dimension_tag == specfem::element::dimension_tag::dim3,
                "assemble_mass_vector takes a dim3 assembly; Tags must be a "
                "dim3 bundle.");

  if (assembly.mesh.element_grid != 5) {
    throw std::runtime_error(
        "specfem::linear_system::assemble_mass_vector: only NGLL == 5 "
        "meshes are supported (the only 3D instantiation).");
  }

  constexpr auto forward = specfem::simulation::field_type::forward;
  constexpr auto outer = specfem::element::mpi_tag::outer;
  constexpr auto inner = specfem::element::mpi_tag::inner;

  auto &field = assembly.fields.template get_simulation_field<forward>();
  const auto &field_impl = field.template get_field<Tags::medium_tag>();
  const auto mass = field_impl.get_mass_inverse();
  const auto h_mass = field_impl.get_host_mass_inverse();

  if (field_impl.nglob != dof_map.nglob()) {
    throw std::runtime_error(
        "specfem::linear_system::assemble_mass_vector: the dof map does not "
        "match the assembly's forward field.");
  }

  // The forward field's (not-yet-inverted) mass storage is the accumulation
  // target of the production path; treat it strictly as scratch.
  Kokkos::deep_copy(mass, 0);

  // dt = 0: the Stacey lumped term (dt/2) C 1 vanishes exactly, leaving the
  // pure mass on any mesh.
  using BaseTags =
      specfem::tags::Tags<Tags::dimension_tag, forward, Tags::medium_tag>;
  specfem::compute::compute_mass_matrix<5,
                                        specfem::tags::expand<BaseTags, outer>>(
      assembly, static_cast<type_real>(0));
  specfem::compute::compute_mass_matrix<5,
                                        specfem::tags::expand<BaseTags, inner>>(
      assembly, static_cast<type_real>(0));

  Kokkos::deep_copy(h_mass, mass);

  auto mass_vector = Teuchos::rcp(new vector_type(dof_map.owned_map()));
  {
    auto view = mass_vector->getLocalViewHost(Tpetra::Access::OverwriteAll);
    for (int iglob = 0; iglob < dof_map.nglob(); ++iglob) {
      for (int icomp = 0; icomp < dof_map.ncomp(); ++icomp) {
        view(static_cast<std::size_t>(dof_map.gid(iglob, icomp)), 0) =
            h_mass(iglob, icomp);
      }
    }
  }

  // Leave the assembly as found for the explicit solver's own mass init.
  Kokkos::deep_copy(mass, 0);
  Kokkos::deep_copy(h_mass, 0);

  return mass_vector;
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
template Teuchos::RCP<specfem::linear_system::vector_type>
specfem::linear_system::assemble_mass_vector<
    specfem::linear_system_impl::elastic_isotropic_tags>(
    specfem::assembly::assembly<specfem::element::dimension_tag::dim3> &,
    const specfem::linear_system::DofMap &);

#endif // SPECFEM_ENABLE_TRILINOS
