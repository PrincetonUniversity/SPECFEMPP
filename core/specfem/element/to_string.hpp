#pragma once

#include "specfem/simulation.hpp"
#include "specfem/utilities/errors.hpp"
#include "tags.hpp"
#include <array>
#include <string>
#include <tuple>

namespace specfem {
namespace element {
/**
 * @brief Convert medium, property, and boundary tags to string.
 *
 * @param medium Medium type
 * @param property_tag Property type
 * @param boundary_tag Boundary condition type
 * @return Combined string representation
 */
const std::string to_string(const medium_tag &medium,
                            const property_tag &property_tag,
                            const boundary_tag &boundary_tag);

/**
 * @brief Convert medium, property, and attenuation tags to string.
 *
 * @param medium Medium type
 * @param property_tag Property type
 * @param attenuation_tag Attenuation type
 * @return Combined string representation
 */
const std::string to_string(const medium_tag &medium,
                            const property_tag &property_tag,
                            const attenuation_tag &attenuation_tag);

/**
 * @brief Convert dimension tag to string.
 *
 * @param dimension Dimension type
 * @return String representation
 */
const std::string to_string(const dimension_tag &dimension);

/**
 * @brief Convert medium tag to string.
 *
 * @param medium Medium type
 * @return String representation
 */
const std::string to_string(const medium_tag &medium);

/**
 * @brief Convert property tag to string.
 *
 * @param property Property type
 * @return String representation
 */
const std::string to_string(const property_tag &property);

/**
 * @brief Convert boundary tag to string.
 *
 * @param boundary Boundary condition type
 * @return String representation
 */
const std::string to_string(const boundary_tag &boundary);

/**
 * @brief Convert attenuation tag to string.
 *
 * @param attenuation Attenuation type
 * @return String representation
 */
const std::string to_string(const attenuation_tag &attenuation);

/**
 * @brief Convert field/wavefield type to string.
 *
 * @param wavefield Wavefield type (forward, backward, adjoint)
 * @return String representation
 */
const std::string to_string(const specfem::simulation::field_type &wavefield);

/**
 * @brief Parse medium tag from string representation.
 *
 * @param medium_tag String representation of medium type
 * @return Corresponding medium_tag enumeration value
 * @throws std::runtime_error if string is not recognized
 */
specfem::element::medium_tag from_string(const std::string &medium_tag);

} // namespace element
} // namespace specfem
