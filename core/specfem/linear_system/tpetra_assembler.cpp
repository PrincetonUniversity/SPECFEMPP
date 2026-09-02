#include "specfem/linear_system/tpetra_assembler.hpp"

#ifdef SPECFEM_ENABLE_TRILINOS

#include "specfem/assembly/assembly.hpp"
#include "specfem/datatype/element_index_range.hpp"
#include "specfem/element/to_string.hpp"
#include "specfem/linear_system/element_stiffness.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <sstream>
#include <stdexcept>

template <typename Tags>
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3)
specfem::linear_system::StiffnessAssembler<Tags>::StiffnessAssembler(
    const AssemblyType &assembly, const FEAssemblyType &fe,
    const int batch_size, const specfem::linear_system::StiffnessScope scope)
    : assembly_(assembly), fe_(fe), batch_size_(batch_size) {

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
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3)
void specfem::linear_system::StiffnessAssembler<Tags>::fill_matrix(
    SparseMatrixView<MappingType> &matrix) const {
  const auto &mapping = matrix.mapping();
  const int npoints = mapping.ngllz() * mapping.nglly() * mapping.ngllx();
  const int ndof_e = ncomp * npoints;

  const auto elements = mapping.elements();
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

    // One block-diagonal update for the whole batch: the dof set names this
    // batch's elements, and the rank-3 block carries one dense element block
    // each. h_k_e is allocated once at the full batch size and reused, so on a
    // partial final batch it is longer than the dof set -- the view consumes
    // the leading batch_count blocks that the probe just wrote.
    const auto dofs =
        mapping(batch, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
    matrix(dofs, dofs) += h_k_e;
  }
}

template <typename Tags>
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3)
Teuchos::RCP<specfem::linear_system::crs_matrix_type>
specfem::linear_system::StiffnessAssembler<Tags>::assemble() const {
  SparseMatrixView<MappingType> matrix(fe_.full_matrix_graph(), fe_.mapping());
  matrix.begin_fill();
  fill_matrix(matrix);
  // Migrates owned+shared contributions into the owned map -- the Export(ADD)
  // that reproduces the matrix-free assembly sum across ranks -- and
  // fill-completes.
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
template class specfem::linear_system::StiffnessAssembler<
    specfem::linear_system_impl::elastic_isotropic_tags>;

#endif // SPECFEM_ENABLE_TRILINOS
