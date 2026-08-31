#include "specfem/linear_system/system_layout.hpp"

#ifdef SPECFEM_ENABLE_TRILINOS

#include "specfem/assembly/assembly.hpp"
#include "specfem/tags.hpp"
#include <Teuchos_ArrayView.hpp>
#include <Tpetra_Core.hpp>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

template <typename Tags>
specfem::linear_system::SystemLayout<Tags>::SystemLayout(
    const AssemblyType &assembly,
    const Teuchos::RCP<const Teuchos::Comm<int>> &comm)
    : assembly_(&assembly), comm_(comm) {

  if (comm_->getSize() > 1) {
    throw std::runtime_error(
        "specfem::linear_system::SystemLayout: distributed assembly on " +
        std::to_string(comm_->getSize()) +
        " ranks is not implemented yet. SPECFEM++ numbers global points "
        "per rank, and the cross-rank GID negotiation is a follow-up of "
        "issue #1982. Run on a single rank.");
  }

  const auto &field = assembly_->fields.template get_simulation_field<
      specfem::simulation::field_type::forward>();
  nglob_ = field.template get_nglob<medium_tag>();
  ncomp_ = specfem::element::attributes<dimension_tag, medium_tag>::components;

  // The layout is the one place that owns the numbering, so the agreement
  // between it and the field it was derived from is checked once here rather
  // than by every consumer.
  const auto &field_impl = field.template get_field<medium_tag>();
  if (field_impl.nglob != nglob_) {
    throw std::runtime_error(
        "specfem::linear_system::SystemLayout: the medium's global point "
        "count disagrees with the assembly's forward field.");
  }

  const Tpetra::global_size_t num_entries =
      static_cast<Tpetra::global_size_t>(num_global_dofs());
  const global_ordinal_type index_base = 0;
  owned_map_ = Teuchos::rcp(new map_type(num_entries, index_base, comm_));
  // At one rank every dof is owned; a distributed build replaces this with
  // a map that additionally holds the shared-interface dofs of neighbor
  // ranks, and the assemblers gain an Export(ADD) into the owned matrix.
  overlap_map_ = owned_map_;

  element_columns_.resize(assembly_->element_types.nspec);
}

template <typename Tags>
specfem::linear_system::SystemLayout<Tags>
specfem::linear_system::SystemLayout<Tags>::from_assembly(
    const AssemblyType &assembly) {
  return SystemLayout(assembly, Tpetra::getDefaultComm());
}

template <typename Tags>
const std::vector<specfem::linear_system::global_ordinal_type> &
specfem::linear_system::SystemLayout<Tags>::element_column_gids(
    const int ispec) const {

  auto &cols = element_columns_[ispec];
  if (!cols.empty()) {
    return cols;
  }

  const auto &field = assembly_->fields.template get_simulation_field<
      specfem::simulation::field_type::forward>();
  const int ngllz = field.ngllz;
  const int nglly = field.nglly;
  const int ngllx = field.ngllx;
  const int npoints = ngllz * nglly * ngllx;

  cols.resize(static_cast<std::size_t>(ncomp_) * npoints);
  for (int iz = 0; iz < ngllz; ++iz) {
    for (int iy = 0; iy < nglly; ++iy) {
      for (int ix = 0; ix < ngllx; ++ix) {
        const int iglob =
            field.template get_iglob<false, medium_tag>(ispec, iz, iy, ix);
        // Same ordering as local_dof_index<NGLL> on the cubic GLL grid.
        const int point = (iz * nglly + iy) * ngllx + ix;
        for (int icomp = 0; icomp < ncomp_; ++icomp) {
          cols[static_cast<std::size_t>(icomp) * npoints + point] =
              gid(iglob, icomp);
        }
      }
    }
  }
  return cols;
}

template <typename Tags>
Teuchos::RCP<const specfem::linear_system::crs_graph_type>
specfem::linear_system::SystemLayout<Tags>::full_graph() const {

  if (!full_graph_.is_null()) {
    return full_graph_;
  }

  const auto &field = assembly_->fields.template get_simulation_field<
      specfem::simulation::field_type::forward>();
  const int ngllz = field.ngllz;
  const int nglly = field.nglly;
  const int ngllx = field.ngllx;
  const int npoints = ngllz * nglly * ngllx;
  const int ndof_e = ncomp_ * npoints;

  const auto elements =
      assembly_->element_types.get_elements_on_host(medium_tag);

  // Pass 1: per-row allocation upper bound. A row interacts with every dof
  // of every element sharing its mesh point; per-element duplicates are
  // inserted raw in pass 2 (fillComplete merges them), so the bound must --
  // and does exactly -- cover the raw insert count.
  std::vector<std::size_t> adjacent(nglob_, 0);
  for (int i = 0; i < elements.size(); ++i) {
    const int ispec = elements(i);
    for (int iz = 0; iz < ngllz; ++iz) {
      for (int iy = 0; iy < nglly; ++iy) {
        for (int ix = 0; ix < ngllx; ++ix) {
          ++adjacent[field.template get_iglob<false, medium_tag>(ispec, iz, iy,
                                                                 ix)];
        }
      }
    }
  }

  std::vector<std::size_t> entries_per_row(
      static_cast<std::size_t>(num_global_dofs()), 0);
  for (int iglob = 0; iglob < nglob_; ++iglob) {
    for (int icomp = 0; icomp < ncomp_; ++icomp) {
      entries_per_row[static_cast<std::size_t>(gid(iglob, icomp))] =
          adjacent[iglob] * static_cast<std::size_t>(ndof_e);
    }
  }

  auto graph = Teuchos::rcp(new crs_graph_type(
      overlap_map_, Teuchos::ArrayView<const std::size_t>(
                        entries_per_row.data(), entries_per_row.size())));

  // Pass 2: every dof of an element couples to every other dof of that
  // element, so each element contributes its full column list to each of its
  // rows.
  for (int i = 0; i < elements.size(); ++i) {
    const auto &cols = element_column_gids(elements(i));
    for (int r = 0; r < ndof_e; ++r) {
      graph->insertGlobalIndices(cols[r], ndof_e, cols.data());
    }
  }

  graph->fillComplete(owned_map_, owned_map_);
  full_graph_ = graph;
  return full_graph_;
}

template <typename Tags>
Teuchos::RCP<specfem::linear_system::crs_matrix_type>
specfem::linear_system::SystemLayout<Tags>::full_matrix() const {
  // Matrix on the fill-complete static graph: values start at zero and only
  // sumIntoGlobalValues is allowed, which is exactly what the assemblers use.
  return Teuchos::rcp(new crs_matrix_type(full_graph()));
}

template <typename Tags>
Teuchos::RCP<specfem::linear_system::crs_matrix_type>
specfem::linear_system::SystemLayout<Tags>::block_diagonal_matrix(
    const std::function<bool(int)> &mask) const {

  const auto admitted = [&mask](const int iglob) {
    return !mask || mask(iglob);
  };

  std::vector<std::size_t> entries_per_row(
      static_cast<std::size_t>(num_global_dofs()), 0);
  for (int iglob = 0; iglob < nglob_; ++iglob) {
    if (!admitted(iglob)) {
      continue;
    }
    for (int r = 0; r < ncomp_; ++r) {
      entries_per_row[static_cast<std::size_t>(gid(iglob, r))] = ncomp_;
    }
  }

  auto graph = Teuchos::rcp(new crs_graph_type(
      overlap_map_, Teuchos::ArrayView<const std::size_t>(
                        entries_per_row.data(), entries_per_row.size())));

  std::vector<global_ordinal_type> cols(ncomp_);
  for (int iglob = 0; iglob < nglob_; ++iglob) {
    if (!admitted(iglob)) {
      continue;
    }
    for (int c = 0; c < ncomp_; ++c) {
      cols[c] = gid(iglob, c);
    }
    for (int r = 0; r < ncomp_; ++r) {
      graph->insertGlobalIndices(gid(iglob, r), ncomp_, cols.data());
    }
  }
  graph->fillComplete(owned_map_, owned_map_);

  return Teuchos::rcp(new crs_matrix_type(graph));
}

template <typename Tags>
Teuchos::RCP<specfem::linear_system::vector_type>
specfem::linear_system::SystemLayout<Tags>::create_vector() const {
  return Teuchos::rcp(new vector_type(owned_map_));
}

template <typename Tags>
void specfem::linear_system::SystemLayout<Tags>::validate_field_extents(
    const host_field_view_type &field, const char *what) const {
  if (static_cast<int>(field.extent(0)) != nglob_ ||
      static_cast<int>(field.extent(1)) != ncomp_) {
    std::ostringstream message;
    message << "specfem::linear_system::SystemLayout::" << what
            << ": host field has shape (" << field.extent(0) << ", "
            << field.extent(1) << "), expected (" << nglob_ << ", " << ncomp_
            << ").";
    throw std::runtime_error(message.str());
  }
}

template <typename Tags>
void specfem::linear_system::SystemLayout<Tags>::scatter(
    const host_field_view_type &src, vector_type &dst) const {
  validate_field_extents(src, "scatter");

  // Element-by-element on purpose: the component-blocked layout happens to
  // make this contiguous today, and exploiting that would tie every consumer
  // to the current ordering (see the class docs).
  auto view = dst.getLocalViewHost(Tpetra::Access::OverwriteAll);
  for (int iglob = 0; iglob < nglob_; ++iglob) {
    for (int icomp = 0; icomp < ncomp_; ++icomp) {
      view(static_cast<std::size_t>(gid(iglob, icomp)), 0) = src(iglob, icomp);
    }
  }
}

template <typename Tags>
Teuchos::RCP<specfem::linear_system::vector_type>
specfem::linear_system::SystemLayout<Tags>::scatter(
    const host_field_view_type &src) const {
  auto dst = create_vector();
  scatter(src, *dst);
  return dst;
}

template <typename Tags>
void specfem::linear_system::SystemLayout<Tags>::gather(
    const vector_type &src, const host_field_view_type &dst) const {
  validate_field_extents(dst, "gather");

  const auto view = src.getLocalViewHost(Tpetra::Access::ReadOnly);
  for (int iglob = 0; iglob < nglob_; ++iglob) {
    for (int icomp = 0; icomp < ncomp_; ++icomp) {
      dst(iglob, icomp) = view(static_cast<std::size_t>(gid(iglob, icomp)), 0);
    }
  }
}

template <typename Tags>
void specfem::linear_system::SystemLayout<Tags>::scatter_point_block(
    crs_matrix_type &matrix, const int iglob,
    const host_field_view_type &block) const {

  if (static_cast<int>(block.extent(0)) != ncomp_ ||
      static_cast<int>(block.extent(1)) != ncomp_) {
    std::ostringstream message;
    message << "specfem::linear_system::SystemLayout::scatter_point_block: "
               "block has shape ("
            << block.extent(0) << ", " << block.extent(1) << "), expected ("
            << ncomp_ << ", " << ncomp_ << ").";
    throw std::runtime_error(message.str());
  }

  std::vector<global_ordinal_type> cols(ncomp_);
  for (int c = 0; c < ncomp_; ++c) {
    cols[c] = gid(iglob, c);
  }

  std::vector<scalar_type> values(ncomp_);
  for (int r = 0; r < ncomp_; ++r) {
    for (int c = 0; c < ncomp_; ++c) {
      values[c] = block(r, c);
    }
    const int updated = matrix.sumIntoGlobalValues(gid(iglob, r), ncomp_,
                                                   values.data(), cols.data());
    if (updated != ncomp_) {
      std::ostringstream message;
      message << "specfem::linear_system::SystemLayout::scatter_point_block: "
                 "block scatter at point "
              << iglob << " updated " << updated << " of " << ncomp_
              << " entries; the matrix graph does not carry this point's "
                 "block.";
      throw std::runtime_error(message.str());
    }
  }
}

template <typename Tags>
bool specfem::linear_system::SystemLayout<Tags>::has_point_block(
    const crs_matrix_type &matrix, const int iglob) const {

  const std::size_t max_entries = matrix.getGlobalMaxNumRowEntries();
  typename crs_matrix_type::nonconst_global_inds_host_view_type indices(
      "specfem::linear_system::point_block_indices", max_entries);
  typename crs_matrix_type::nonconst_values_host_view_type values(
      "specfem::linear_system::point_block_values", max_entries);

  for (int r = 0; r < ncomp_; ++r) {
    std::size_t num_entries = 0;
    matrix.getGlobalRowCopy(gid(iglob, r), indices, values, num_entries);
    for (int c = 0; c < ncomp_; ++c) {
      const auto column = gid(iglob, c);
      bool found = false;
      for (std::size_t k = 0; k < num_entries; ++k) {
        if (indices(k) == column) {
          found = true;
          break;
        }
      }
      if (!found) {
        return false;
      }
    }
  }
  return true;
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
template class specfem::linear_system::SystemLayout<
    specfem::linear_system_impl::elastic_isotropic_tags>;

#endif // SPECFEM_ENABLE_TRILINOS
