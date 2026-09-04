#pragma once

#ifdef SPECFEM_ENABLE_TRILINOS

#include "specfem/enums.hpp"
#include "specfem/linear_system/dof_map.hpp"
#include <Teuchos_RCP.hpp>

namespace specfem {
namespace linear_system {

/**
 * @brief Assemble the lumped diagonal mass matrix \f$ M \f$ of one medium as
 * a `Tpetra::Vector`.
 *
 * Drives the production `initialize_mass_matrix` accumulation path with
 * `dt = 0`: the Stacey boundary contribution to the lumped mass is exactly
 * \f$ (\Delta t / 2) \, C \, \mathbf{1} \f$ (linear in `dt`), so it vanishes
 * identically and the result is the pure mass \f$ M \f$ on any mesh,
 * including meshes with Stacey boundaries. The damping matrix \f$ C \f$ is
 * assembled separately (see @ref DampingAssembler).
 *
 * The forward simulation field's mass storage is used as accumulation
 * scratch: it is zeroed before the accumulation, read back through the host
 * mirror, and zeroed again on exit, so the assembly is left as found. The
 * inversion step of the explicit solver (`invert_mass`) is never called.
 *
 * Entry `DofMap::gid(iglob, icomp)` of the returned vector holds the lumped
 * mass of component `icomp` at mesh point `iglob`.
 *
 * @tparam Tags Compile-time tags (dimension, medium, property, attenuation);
 *              dimension must be `dim3`; only `dim3, elastic, isotropic,
 *              none` is instantiated
 * @param assembly Assembled mesh, material properties, and fields; the
 *        forward field's mass storage is used as scratch
 * @param dof_map Dof numbering shared with the stiffness/damping matrices
 * @return Lumped mass vector on the owned map
 */
template <typename Tags>
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3)
Teuchos::RCP<vector_type>
assemble_mass_vector(specfem::assembly::assembly<Tags::dimension_tag> &assembly,
                     const DofMap &dof_map);

} // namespace linear_system
} // namespace specfem

#endif // SPECFEM_ENABLE_TRILINOS
