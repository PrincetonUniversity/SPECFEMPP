#pragma once

#include "specfem/enums.hpp"
#include "specfem/mesh.hpp"

#include <Kokkos_Core.hpp>
#include <array>
#include <fstream>
#include <tuple>

namespace specfem::io::mesh::impl::fortran::dim3_impl {

/**
 * @brief Voigt stiffnesses in the order taken by the anisotropic material
 * constructor: \f$c_{11}, c_{12}, c_{13}, c_{14}, c_{15}, c_{16}, c_{22},
 * \ldots, c_{66}\f$.
 */
using stiffness_array = std::array<type_real, 21>;

/**
 * @brief Build the Voigt stiffness matrix for a MESHFEM3D anisotropy flag.
 *
 * Port of SPECFEM3D's `model_aniso` (`src/generate_databases/model_aniso.f90`).
 * The MESHFEM3D database carries only the anisotropy flag alongside
 * \f$(\rho, v_p, v_s)\f$, so the 21 stiffnesses are reconstructed here from
 * the transversely-isotropic Love parameters
 *
 * \f[
 * A = \rho v_{ph}^2, \quad C = \rho v_{pv}^2, \quad N = \rho v_{sh}^2,
 * \quad L = \rho v_{sv}^2, \quad F = \eta (A - 2L),
 * \f]
 *
 * perturbed by the zeta-dependent terms of Chen & Tromp (2006, appendix A)
 * and rotated from the geographic frame (South, East, up) to the Cartesian
 * frame used by the solver (East, North, up).
 *
 * @param iflag_aniso Anisotropy flag from the database: `<= 0` reduces to the
 * isotropic-equivalent matrix, `1` and `2` select the two perturbation models.
 * @param rho Density.
 * @param vp Compressional wave speed.
 * @param vs Shear wave speed.
 * @return The 21 independent Voigt stiffnesses.
 *
 * @throws std::runtime_error If @p iflag_aniso names an unsupported model.
 */
stiffness_array model_aniso(const int iflag_aniso, const type_real rho,
                            const type_real vp, const type_real vs);

} // namespace specfem::io::mesh::impl::fortran::dim3_impl

namespace specfem::io::mesh::impl::fortran::dim3 {

/**
 * @brief Read material properties and control node indices from MESHFEM3D
 * database file
 *
 * Reads material property data and control node index mapping from a binary
 * MESHFEM3D database file for 3D spectral element simulations.
 *
 * @param stream Input file stream positioned at the materials section
 * @param ngnod Number of control nodes per spectral element (e.g., 8 for
 * hexahedral elements)
 *
 * @return std::tuple containing:
 *         - Number of spectral elements in the mesh
 *         - Number of control nodes in the zeta direction
 *         - Number of control nodes in the eta direction
 *         - Number of control nodes in the xi direction
 *         - Control node indices array mapping spectral elements to materials
 *         - Materials object containing material specifications and
 * classifications
 *
 * @throws std::runtime_error If file reading fails or invalid material data is
 * encountered
 */
std::tuple<int, int, int, int,
           Kokkos::View<int **, Kokkos::LayoutLeft, Kokkos::HostSpace>,
           specfem::mesh::materials<specfem::element::dimension_tag::dim3>>
read_materials(std::ifstream &stream, const int ngnod,
               const bool attenuation_enabled);

} // namespace specfem::io::mesh::impl::fortran::dim3
