#pragma once

#include "enumerations/interface.hpp"
#include "specfem/mesh.hpp"

#include <Kokkos_Core.hpp>
#include <fstream>
#include <tuple>

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
           specfem::mesh::materials<specfem::dimension::type::dim3> >
read_materials(std::ifstream &stream, const int ngnod);

} // namespace specfem::io::mesh::impl::fortran::dim3
