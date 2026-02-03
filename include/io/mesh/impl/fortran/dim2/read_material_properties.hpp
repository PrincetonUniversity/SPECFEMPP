#pragma once

#include "enumerations/dimension.hpp"
#include "specfem/mesh.hpp"

#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

namespace specfem {
namespace io {
namespace mesh {
namespace impl {
namespace fortran {
namespace dim2 {

/**
 * Read material properties from a fotran binary database
 *
 * @param stream Stream object for fortran binary file buffered to materials
 * section
 * @param numat Number of materials to be read
 * @return std::vector<specfem::medium *> Pointer to material objects read
 * from the database file
 */

specfem::mesh::materials<specfem::dimension::type::dim2>
read_material_properties(
    std::ifstream &stream, const int numat, const int nspec,
    const specfem::enums::elastic_wave wave,
    const specfem::enums::electromagnetic_wave electromagnetic_wave,
    const specfem::kokkos::HostView2d<int> knods);

} // namespace dim2
} // namespace fortran
} // namespace impl
} // namespace mesh
} // namespace io
} // namespace specfem
