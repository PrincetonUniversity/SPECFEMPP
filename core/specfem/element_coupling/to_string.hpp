#include "specfem/element_coupling.hpp"

namespace specfem::element_coupling {

/**
 * @brief Convert interface tag to string.
 * @param interface_tag Interface coupling type
 * @return String representation ("elastic_acoustic" or "acoustic_elastic")
 */
std::string to_string(const interface_tag &interface_tag);

/**
 * @brief Convert flux scheme tag to string.
 *
 * @param flux_scheme_tag
 * @return String representation
 */
std::string to_string(const flux_scheme_tag &flux_scheme_tag);

std::ostream &
operator<<(std::ostream &stream,
           const specfem::element_coupling::interface_tag &interface_tag);
std::ostream &
operator<<(std::ostream &stream,
           const specfem::element_coupling::flux_scheme_tag &flux_scheme_tag);

} // namespace specfem::element_coupling
