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
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3)
specfem::linear_system::StiffnessAssembler<Tags>::StiffnessAssembler(
    const AssemblyType &assembly, const int batch_size,
    const specfem::linear_system::StiffnessScope scope)
    : assembly_(assembly), dof_map_(assembly, Tags{}), batch_size_(batch_size) {

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
std::vector<specfem::linear_system::global_ordinal_type>
specfem::linear_system::StiffnessAssembler<Tags>::element_column_gids(
    const int ispec) const {
  const auto &field = assembly_.fields.template get_simulation_field<
      specfem::simulation::field_type::forward>();
  const int ngllz = field.ngllz;
  const int nglly = field.nglly;
  const int ngllx = field.ngllx;
  const int npoints = ngllz * nglly * ngllx;

  std::vector<global_ordinal_type> cols(static_cast<std::size_t>(ncomp) *
                                        npoints);
  for (int iz = 0; iz < ngllz; ++iz) {
    for (int iy = 0; iy < nglly; ++iy) {
      for (int ix = 0; ix < ngllx; ++ix) {
        const int iglob =
            field.template get_iglob<false, medium_tag>(ispec, iz, iy, ix);
        // Same ordering as local_dof_index<NGLL> on the cubic GLL grid.
        const int point = (iz * nglly + iy) * ngllx + ix;
        for (int icomp = 0; icomp < ncomp; ++icomp) {
          cols[static_cast<std::size_t>(icomp) * npoints + point] =
              dof_map_.gid(iglob, icomp);
        }
      }
    }
  }
  return cols;
}

template <typename Tags>
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3)
Teuchos::RCP<const specfem::linear_system::crs_graph_type>
specfem::linear_system::StiffnessAssembler<Tags>::build_graph() const {
  const auto elements =
      assembly_.element_types.get_elements_on_host(medium_tag);

  // Every dof of an element couples to every other dof of that element, so
  // the stiffness sparsity is one dense coupling block per element. The dof
  // map turns that description into the library's graph; per-element
  // duplicates at shared mesh points are inserted raw and merged by
  // fillComplete, and the allocation bound it derives (block size per block
  // a row appears in) covers the raw insert count exactly.
  return dof_map_.build_graph([&](const auto &visit) {
    for (int i = 0; i < elements.size(); ++i) {
      visit(element_column_gids(elements(i)));
    }
  });
}

template <typename Tags>
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3)
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
      const auto cols = element_column_gids(ispec);
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
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3)
Teuchos::RCP<specfem::linear_system::crs_matrix_type>
specfem::linear_system::StiffnessAssembler<Tags>::assemble() const {
  const auto graph = build_graph();

  // Matrix on the fill-complete static graph: values start at zero and only
  // sumIntoGlobalValues is allowed, which is exactly what the scatter uses.
  auto matrix = Teuchos::rcp(new crs_matrix_type(graph));

  fill_matrix(*matrix);

  matrix->fillComplete(dof_map_.owned_map(), dof_map_.owned_map());

  // MPI-ready seam: a distributed build fills the matrix on the overlap map
  // and Export(ADD)s into the owned map here, reproducing the matrix-free
  // assembly sum across ranks. Unreachable at one rank (DofMap rejects
  // larger communicators), where overlap == owned and the matrix is final.
  if (!dof_map_.overlap_map()->isSameAs(*dof_map_.owned_map())) {
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
