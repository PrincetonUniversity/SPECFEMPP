#pragma once

#include "enumerations/interface.hpp"
#include "specfem/mesh.hpp"

namespace specfem {
namespace io {
namespace mesh {
namespace impl {
namespace fortran {
namespace dim2 {

/**
 * @brief Read tangential elements from mesh database
 *
 * @param stream Input stream
 * @param nnodes_tangential_curve Number of nodes on the tangential curve
 * @return specfem::mesh::elements::tangential_elements
 *
 */
specfem::mesh::elements::tangential_elements<specfem::dimension::type::dim2>
read_tangential_elements(std::ifstream &stream,
                         const int nnodes_tangential_curve);

/**
 * @brief Read axial elements from mesh database
 *
 * @param stream Input stream
 * @param nelem_on_the_axis Number of elements on the axis
 * @param nspec Number of spectral elements
 * @return specfem::mesh::elements::axial_elements
 *
 */
specfem::mesh::elements::axial_elements<specfem::dimension::type::dim2>
read_axial_elements(std::ifstream &stream, const int nelem_on_the_axis,
                    const int nspec);

} // namespace dim2
} // namespace fortran
} // namespace impl
} // namespace mesh
} // namespace io
} // namespace specfem
