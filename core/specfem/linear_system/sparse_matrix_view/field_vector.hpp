#pragma once

#ifdef SPECFEM_ENABLE_TRILINOS

#include "specfem/linear_system/tpetra_types.hpp"
#include <Kokkos_Core.hpp>
#include <Tpetra_Access.hpp>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>

namespace specfem {
namespace linear_system_impl {

/**
 * @brief Whether a field and a solver vector are the same bytes in the same
 * order, so that the copy between them is one `deep_copy`.
 *
 * Both must be contiguous, and the mapping is probed -- not assumed -- to be
 * component-blocked with unit stride in `iglob`.
 *
 * @tparam MappingType Dof numbering
 * @tparam FieldView Rank-2 host field view, `(nglob, ncomp)`
 * @tparam LocalView Rank-2 host view of the vector, `(num_global_dofs, 1)`
 * @param mapping Dof numbering
 * @param field Field storage
 * @param local Vector storage
 * @return `true` if a flat copy is exact
 */
template <typename MappingType, typename FieldView, typename LocalView>
bool field_matches_vector_layout(const MappingType &mapping,
                                 const FieldView &field,
                                 const LocalView &local) {
  const int nglob = mapping.nglob();
  const int ncomp = mapping.ncomp();

  // Too small to probe the strides below; the indexed path is exact anyway.
  if (nglob < 2 || ncomp < 1) {
    return false;
  }

  if (!field.span_is_contiguous() || local.stride(0) != 1) {
    return false;
  }

  // Component-blocked with unit stride in iglob, asked of the mapping rather
  // than assumed.
  if (mapping(0, 0) != 0 || mapping(1, 0) != 1) {
    return false;
  }
  if (ncomp > 1 && mapping(0, 1) != nglob) {
    return false;
  }

  return true;
}

/// Throw unless a field and a vector have the extents `mapping` describes
template <typename MappingType, typename FieldView, typename LocalView>
void check_field_vector_extents(const MappingType &mapping,
                                const FieldView &field, const LocalView &local,
                                const std::string &what) {
  const auto expected_dofs =
      static_cast<std::size_t>(mapping.num_global_dofs());

  if (static_cast<int>(field.extent(0)) != mapping.nglob() ||
      static_cast<int>(field.extent(1)) != mapping.ncomp()) {
    std::ostringstream message;
    message << "specfem::linear_system::" << what
            << ": the mapping describes a " << mapping.nglob() << "x"
            << mapping.ncomp() << " field, but the field passed is "
            << field.extent(0) << "x" << field.extent(1) << ".";
    throw std::runtime_error(message.str());
  }

  if (static_cast<std::size_t>(local.extent(0)) != expected_dofs) {
    std::ostringstream message;
    message << "specfem::linear_system::" << what << ": the mapping describes "
            << expected_dofs << " degrees of freedom, but the vector holds "
            << local.extent(0) << ".";
    throw std::runtime_error(message.str());
  }
}

} // namespace linear_system_impl

namespace linear_system {

/**
 * @brief Copy a SPECFEM++ field into a solver vector.
 *
 * specfem::linear_system::Mapping numbers dofs so that `gid(iglob, icomp)` is
 * the `Kokkos::LayoutLeft` offset of `(iglob, icomp)`, which is how a field is
 * stored, so the copy is a single `deep_copy`. Falls back to an indexed loop
 * when that layout cannot be confirmed at run time.
 *
 * @tparam MappingType Dof numbering; see specfem::linear_system::Mapping
 * @tparam FieldView Rank-2 host field view, `(nglob, ncomp)`
 * @param mapping Dof numbering shared by the field and the vector
 * @param field Source field storage (host)
 * @param vector Destination vector; fully overwritten
 *
 * @throws std::runtime_error if the extents disagree with `mapping`
 */
template <typename MappingType, typename FieldView>
void copy_field_to_vector(const MappingType &mapping, const FieldView &field,
                          vector_type &vector) {
  auto local = vector.getLocalViewHost(Tpetra::Access::OverwriteAll);

  specfem::linear_system_impl::check_field_vector_extents(
      mapping, field, local, "copy_field_to_vector");

  const int nglob = mapping.nglob();
  const int ncomp = mapping.ncomp();

  if (specfem::linear_system_impl::field_matches_vector_layout(mapping, field,
                                                               local)) {
    const auto span = static_cast<std::size_t>(nglob) * ncomp;
    const Kokkos::View<const typename FieldView::value_type *,
                       typename FieldView::memory_space,
                       Kokkos::MemoryUnmanaged>
        source(field.data(), span);
    const Kokkos::View<scalar_type *, typename decltype(local)::memory_space,
                       Kokkos::MemoryUnmanaged>
        destination(local.data(), span);
    Kokkos::deep_copy(destination, source);
    return;
  }

  for (int icomp = 0; icomp < ncomp; ++icomp) {
    for (int iglob = 0; iglob < nglob; ++iglob) {
      local(static_cast<std::size_t>(mapping(iglob, icomp)), 0) =
          field(iglob, icomp);
    }
  }
}

/**
 * @brief Copy a solver vector back into a SPECFEM++ field.
 *
 * The inverse of @ref copy_field_to_vector, with the same fast path and
 * fallback.
 *
 * @tparam MappingType Dof numbering; see specfem::linear_system::Mapping
 * @tparam FieldView Rank-2 host field view, `(nglob, ncomp)`
 * @param mapping Dof numbering shared by the field and the vector
 * @param vector Source vector
 * @param field Destination field storage (host); a Kokkos view's constness is
 *        shallow, so this is written through
 *
 * @throws std::runtime_error if the extents disagree with `mapping`
 */
template <typename MappingType, typename FieldView>
void copy_vector_to_field(const MappingType &mapping, const vector_type &vector,
                          const FieldView &field) {
  const auto local = vector.getLocalViewHost(Tpetra::Access::ReadOnly);

  specfem::linear_system_impl::check_field_vector_extents(
      mapping, field, local, "copy_vector_to_field");

  const int nglob = mapping.nglob();
  const int ncomp = mapping.ncomp();

  if (specfem::linear_system_impl::field_matches_vector_layout(mapping, field,
                                                               local)) {
    const auto span = static_cast<std::size_t>(nglob) * ncomp;
    const Kokkos::View<const scalar_type *,
                       typename decltype(local)::memory_space,
                       Kokkos::MemoryUnmanaged>
        source(local.data(), span);
    const Kokkos::View<typename FieldView::value_type *,
                       typename FieldView::memory_space,
                       Kokkos::MemoryUnmanaged>
        destination(field.data(), span);
    Kokkos::deep_copy(destination, source);
    return;
  }

  for (int icomp = 0; icomp < ncomp; ++icomp) {
    for (int iglob = 0; iglob < nglob; ++iglob) {
      field(iglob, icomp) =
          local(static_cast<std::size_t>(mapping(iglob, icomp)), 0);
    }
  }
}

} // namespace linear_system
} // namespace specfem

#endif // SPECFEM_ENABLE_TRILINOS
