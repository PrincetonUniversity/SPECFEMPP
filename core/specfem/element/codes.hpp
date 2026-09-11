#pragma once

#include "tags.hpp"

namespace specfem {
namespace element {

/**
 * @brief Integer codes for element tags as written by the 3-D mesh databases.
 *
 * The Cartesian and globe mesh databases classify elements with small
 * integers. The code ↔ tag mapping is defined here and nowhere else; readers
 * translate through these functions instead of comparing against literals.
 *
 * - region:   1 = crust_mantle, 2 = outer_core, 3 = inner_core
 * - medium:   1 = acoustic, 2 = elastic
 * - property: 0 = isotropic, 1 = anisotropic
 */

/**
 * @brief Region tag for a database region code.
 * @param code Database region code
 * @return Corresponding region tag
 * @throws std::runtime_error if the code is not a region code
 */
region_tag region_tag_from_code(int code);

/**
 * @brief Medium tag for a database medium code.
 * @param code Database medium code
 * @return Corresponding medium tag
 * @throws std::runtime_error if the code is not a medium code
 */
medium_tag medium_tag_from_code(int code);

/**
 * @brief Property tag for a database property code.
 * @param code Database property code
 * @return Corresponding property tag
 * @throws std::runtime_error if the code is not a property code
 */
property_tag property_tag_from_code(int code);

/**
 * @brief Database code of a region tag.
 * @param region Region tag
 * @return Database region code
 */
int to_code(const region_tag &region);

/**
 * @brief Database code of a medium tag.
 * @param medium Medium tag
 * @return Database medium code
 * @throws std::runtime_error if the medium has no database code
 */
int to_code(const medium_tag &medium);

/**
 * @brief Database code of a property tag.
 * @param property Property tag
 * @return Database property code
 * @throws std::runtime_error if the property has no database code
 */
int to_code(const property_tag &property);

} // namespace element
} // namespace specfem
