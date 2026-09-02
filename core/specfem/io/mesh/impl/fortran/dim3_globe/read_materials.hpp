#pragma once

#include "specfem/mesh.hpp"

#include <fstream>
#include <vector>

namespace specfem::io::mesh::impl::fortran::dim3_globe {

/** @brief Material tags read from a globe database element section. */
struct material_tags {
  std::vector<int> medium_tags;
  std::vector<int> property_tags;
};

/**
 * @brief Read element contexts and material tags from a globe database.
 *
 * @param stream Input stream positioned at the element section
 * @param mesh Globe mesh being populated
 * @return Medium and property tags needed to build material mappings
 */
material_tags read_material_tags(std::ifstream &stream,
                                 specfem::mesh::globe3d_mesh &mesh);

/** @brief Build raw mesh material mappings from globe medium/property tags. */
specfem::mesh::materials<specfem::element::dimension_tag::dim3>
make_materials(const std::vector<int> &medium_tags,
               const std::vector<int> &property_tags,
               const bool attenuation_enabled);

} // namespace specfem::io::mesh::impl::fortran::dim3_globe
