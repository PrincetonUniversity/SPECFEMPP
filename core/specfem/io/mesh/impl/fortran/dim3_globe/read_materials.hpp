#pragma once

#include "specfem/mesh.hpp"

#include <fstream>
#include <vector>

namespace specfem::io::mesh::impl::fortran::dim3_globe {

/**
 * @brief Raw material-classification tags from the globe element section.
 *
 * These tags classify each element well enough to build SPECFEM++ raw mesh
 * material mappings. They are not the final pointwise GLL material properties;
 * those are evaluated later from @c globe_element_context and reference
 * coordinates by the globe model oracle.
 */
struct material_tags {
  /** @brief Globe medium tag for each element, e.g. acoustic or elastic. */
  std::vector<int> medium_tags;

  /** @brief Globe property tag for each element, e.g. isotropic/TISO marker. */
  std::vector<int> property_tags;
};

/**
 * @brief Read element contexts and material tags from a thin globe database.
 *
 * The stream must be positioned at the element count. This function reads the
 * per-element region, medium, property, radial bounds, doubling, and
 * crust/mantle flags; stores evaluator context in @c
 * mesh.globe.element_context; and returns the raw medium/property tags used to
 * initialize @c mesh.materials. The element count is also stored in @c
 * mesh.nspec.
 *
 * @param stream Input stream positioned at the element metadata section
 * @param mesh Globe mesh whose element count and evaluator context are set
 * @return Medium and property tags needed to build material mappings
 * @throws std::runtime_error if the element count is non-positive
 */
material_tags read_material_tags(std::ifstream &stream,
                                 specfem::mesh::globe3d_mesh &mesh);

/**
 * @brief Build raw mesh material mappings from globe medium/property tags.
 *
 * The constructed materials use placeholder constants only to classify element
 * medium, property, and attenuation tags for the generic assembly path. Globe
 * simulations replace the pointwise GLL properties during deferred property
 * setup by querying the globe evaluator.
 *
 * @param medium_tags Per-element globe medium tags from @ref read_material_tags
 * @param property_tags Per-element globe property tags from
 *        @ref read_material_tags
 * @param attenuation_enabled Whether elastic elements should be marked as
 *        constant-isotropic attenuating elements
 * @return 3-D raw mesh material table and per-element material index mapping
 * @throws std::runtime_error if unsupported medium or property tags are present
 */
specfem::mesh::materials<specfem::element::dimension_tag::dim3>
make_materials(const std::vector<int> &medium_tags,
               const std::vector<int> &property_tags,
               const bool attenuation_enabled);

} // namespace specfem::io::mesh::impl::fortran::dim3_globe
