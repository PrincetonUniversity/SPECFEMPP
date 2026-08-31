#include "specfem/linear_system/tpetra_assembler.hpp"

#ifdef SPECFEM_ENABLE_TRILINOS

#include "specfem/assembly/assembly.hpp"
#include "specfem/datatype/element_index_range.hpp"
#include "specfem/element/to_string.hpp"
#include "specfem/linear_system/element_stiffness.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <vector>

template <typename Tags>
specfem::linear_system::StiffnessAssembler<Tags>::StiffnessAssembler(
    const AssemblyType &assembly, const int batch_size,
    const specfem::linear_system::StiffnessScope scope)
    : assembly_(assembly), layout_(SystemLayout<Tags>::from_assembly(assembly)),
      batch_size_(batch_size) {

  if (batch_size_ < 1) {
    throw std::runtime_error(
        "specfem::linear_system::StiffnessAssembler: batch_size must be at "
        "least 1.");
  }

  specfem::linear_system::validate_stiffness_scope<Tags>(assembly_, scope);

  // Single-medium milestone: matrix blocks coupling different media
  // (fluid-solid) are deferred, so reject mixed meshes outright rather than
  // silently assembling an operator that ignores the coupling.
  const int nspec = assembly_.element_types.nspec;
  for (int ispec = 0; ispec < nspec; ++ispec) {
    const auto element_medium = assembly_.element_types.get_medium_tag(ispec);
    if (element_medium != medium_tag) {
      std::ostringstream message;
      message << "specfem::linear_system::StiffnessAssembler: element " << ispec
              << " has medium '" << specfem::element::to_string(element_medium)
              << "'; only single-medium '"
              << specfem::element::to_string(medium_tag)
              << "' meshes are supported. Fluid-solid coupling blocks are "
                 "deferred (issue #1982).";
      throw std::runtime_error(message.str());
    }
  }
}

template <typename Tags>
void specfem::linear_system::StiffnessAssembler<Tags>::fill_matrix(
    crs_matrix_type &matrix) const {
  const auto &field = assembly_.fields.template get_simulation_field<
      specfem::simulation::field_type::forward>();
  const int npoints = field.ngllz * field.nglly * field.ngllx;
  const int ndof_e = ncomp * npoints;

  const auto elements =
      assembly_.element_types.get_elements_on_host(medium_tag);
  const int nelements = elements.size();

  // One block buffer reused across batches; only batch_size_ dense element
  // blocks ever exist at a time -- no global dense matrix.
  Kokkos::View<type_real ***, Kokkos::LayoutRight,
               Kokkos::DefaultExecutionSpace>
      k_e("specfem::linear_system::element_stiffness_blocks",
          std::min(batch_size_, std::max(nelements, 1)), ndof_e, ndof_e);
  auto h_k_e = Kokkos::create_mirror_view(k_e);

  for (int offset = 0; offset < nelements; offset += batch_size_) {
    const int batch_count = std::min(batch_size_, nelements - offset);
    const auto batch = specfem::datatype::subview(
        elements, Kokkos::pair<int, int>(offset, offset + batch_count));

    specfem::linear_system::compute_element_stiffness<Tags>(assembly_, batch,
                                                            k_e);
    Kokkos::deep_copy(h_k_e, k_e);

    for (int e = 0; e < batch_count; ++e) {
      const int ispec = batch(e);
      const auto &cols = layout_.element_column_gids(ispec);
      for (int r = 0; r < ndof_e; ++r) {
        const int updated = matrix.sumIntoGlobalValues(
            cols[r], ndof_e, &h_k_e(e, r, 0), cols.data());
        if (updated != ndof_e) {
          std::ostringstream message;
          message << "specfem::linear_system::StiffnessAssembler: row "
                     "scatter of element "
                  << ispec << " updated " << updated << " of " << ndof_e
                  << " entries; the graph does not match the element "
                     "connectivity.";
          throw std::runtime_error(message.str());
        }
      }
    }
  }
}

template <typename Tags>
Teuchos::RCP<specfem::linear_system::crs_matrix_type>
specfem::linear_system::StiffnessAssembler<Tags>::assemble() const {
  // Zero-valued matrix on the layout's cached full graph: only
  // sumIntoGlobalValues is allowed, which is exactly what the scatter uses.
  auto matrix = layout_.full_matrix();

  fill_matrix(*matrix);

  matrix->fillComplete(layout_.owned_map(), layout_.owned_map());

  // MPI-ready seam: a distributed build fills the matrix on the overlap map
  // and Export(ADD)s into the owned map here, reproducing the matrix-free
  // assembly sum across ranks. Unreachable at one rank (SystemLayout rejects
  // larger communicators), where overlap == owned and the matrix is final.
  if (!layout_.overlap_map()->isSameAs(*layout_.owned_map())) {
    throw std::runtime_error(
        "specfem::linear_system::StiffnessAssembler: overlap map differs "
        "from owned map, but the distributed Export(ADD) step is not "
        "implemented yet (follow-up of issue #1982).");
  }

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
template class specfem::linear_system::StiffnessAssembler<
    specfem::linear_system_impl::elastic_isotropic_tags>;

#endif // SPECFEM_ENABLE_TRILINOS
