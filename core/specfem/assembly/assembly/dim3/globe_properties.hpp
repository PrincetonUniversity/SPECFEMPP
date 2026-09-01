#pragma once

#include "specfem/assembly/assembly.hpp"
#include "specfem/mesh.hpp"

namespace specfem::assembly::dim3_impl {

/** @brief Populate GLL properties by querying the globe model evaluator. */
void read_globe_properties(
    const specfem::mesh::globe3d_mesh &mesh,
    specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly);

} // namespace specfem::assembly::dim3_impl
