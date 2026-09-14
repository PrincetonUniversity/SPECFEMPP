#pragma once

#include "specfem/assembly/assembly.hpp"
#include "specfem/element.hpp"
#include "specfem/enums.hpp"

namespace specfem {
namespace linear_system {

/**
 * @brief Degree-of-freedom numbering of one medium: how a SPECFEM++
 * `(iglob, icomp)` pair becomes a single global dof id.
 *
 * The GID layout is component-blocked: `gid = icomp * nglob + iglob`,
 * deliberately matching SPECFEM++ field storage
 * (`Kokkos::View<type_real **, Kokkos::LayoutLeft>` of shape
 * `(nglob, ncomp)`), so a solver vector maps 1:1 onto field memory with no
 * permutation at solve time. It also matches the element-local ordering
 * `specfem::linear_system::local_dof_index`. The layout lives ONLY in
 * @ref gid -- change it there to swap the ordering.
 *
 * One instance describes one medium: `nglob` and `ncomp` are per-medium
 * quantities (a future multi-medium system holds one numbering and one
 * matrix block per medium).
 *
 * This is the linear-algebra-library-independent half of @ref DofMap: it
 * holds SPECFEM++ quantities only and pulls in no solver headers, so the
 * numbering can be reused unchanged if the Tpetra backend is ever swapped
 * out. The library-specific maps and graphs live in
 * @ref TpetraConnections.
 *
 * @tparam GlobalOrdinal Integer type of the global dof ids; supplied by the
 *                       linear-algebra backend (`Tpetra::Map<>::
 *                       global_ordinal_type` for the Tpetra composition)
 */
template <typename GlobalOrdinal> class DofNumbering {
public:
  /// Integer type of the global dof ids
  using global_ordinal_type = GlobalOrdinal;

  /// Empty numbering: zero points, zero components
  DofNumbering() = default;

  /**
   * @brief Number `nglob` mesh points with `ncomp` components per point.
   *
   * @param nglob Number of unique global mesh points of the medium
   * @param ncomp Number of field components per point (3 for 3D elastic)
   */
  DofNumbering(const int nglob, const int ncomp)
      : nglob_(nglob), ncomp_(ncomp) {}

  /**
   * @brief Number the degrees of freedom of one medium of an assembled
   * simulation.
   *
   * Reads `nglob` from the forward simulation field and the component count
   * from the element attributes.
   *
   * The tag bundle is a value argument rather than an explicit template
   * argument because a constructor cannot be given explicit template
   * arguments, and the medium is not deducible from `assembly`: an assembly
   * is templated on the dimension alone and holds the elements of every
   * medium, while one numbering describes one medium.
   *
   * @tparam Tags Compile-time tags (dimension, medium, property,
   *              attenuation); dimension must be `dim3`
   * @param assembly Assembly with constructed fields
   * @param tags Tag bundle selecting the medium; pass `Tags{}`
   */
  template <typename Tags>
    requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3)
  DofNumbering(const specfem::assembly::assembly<Tags::dimension_tag> &assembly,
               Tags tags)
      : DofNumbering(
            assembly.fields
                .template get_simulation_field<
                    specfem::simulation::field_type::forward>()
                .template get_nglob<Tags::medium_tag>(),
            specfem::element::attributes<Tags::dimension_tag,
                                         Tags::medium_tag>::components) {}

  /**
   * @brief Global id of component `icomp` at mesh point `iglob`.
   *
   * Component-blocked layout `gid = icomp * nglob + iglob` -- the single
   * source of truth for the global dof ordering (see the class docs).
   *
   * @param iglob Per-medium global point index in `[0, nglob())`
   * @param icomp Field component in `[0, ncomp())`
   * @return Global dof id in `[0, num_global_dofs())`
   */
  inline global_ordinal_type gid(const int iglob, const int icomp) const {
    return static_cast<global_ordinal_type>(icomp) * nglob_ + iglob;
  }

  /// Number of unique global mesh points of the medium
  inline int nglob() const { return nglob_; }

  /// Number of field components per mesh point
  inline int ncomp() const { return ncomp_; }

  /// Total number of degrees of freedom: `ncomp * nglob`
  inline global_ordinal_type num_global_dofs() const {
    return static_cast<global_ordinal_type>(ncomp_) * nglob_;
  }

private:
  int nglob_ = 0; ///< Points of the medium
  int ncomp_ = 0; ///< Components per point
};

} // namespace linear_system
} // namespace specfem
