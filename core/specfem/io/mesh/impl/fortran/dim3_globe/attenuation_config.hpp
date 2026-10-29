#pragma once

#include "specfem/attenuation/setup.hpp"
#include "specfem/globe/model_config.hpp"
#include "specfem/mesh/attenuation_config.hpp"

#include <vector>

namespace specfem::io::mesh::impl::fortran::dim3_globe {

/**
 * @brief Reconcile the database attenuation settings with the runtime setup.
 *
 * In a globe run Q comes from the model oracle, which is initialized from the
 * database. The database and the runtime configuration therefore have to
 * agree:
 *
 * - Runtime attenuation on with a database meshed without it throws, because
 *   the oracle returns no Q in that case.
 * - Database attenuation on with runtime attenuation off logs a warning and
 *   runs elastic.
 * - With both on, the runtime band must equal the database band
 *   \f$[1/T_{max}, 1/T_{min}]\f$ (the band the oracle and the mesher used),
 *   an explicit reference frequency is required, and the
 *   @c ATTENUATION_3D_BERKELEY and @c ATTENUATION_GLL model flags are
 *   rejected until their Q output has been validated.
 *
 * The returned configuration disables bulk attenuation, matching
 * SPECFEM3D_GLOBE's Q_mu-only implementation.
 *
 * @param model_config Model selection read from the database
 * @param model_flags The database's model flags
 * @param attenuation_setup Attenuation settings from the runtime configuration
 * @return Attenuation configuration for the mesh (disabled unless both the
 *         database and the runtime configuration enable attenuation)
 * @throws std::runtime_error on any of the inconsistencies listed above
 */
specfem::mesh::attenuation_config
make_attenuation_config(const specfem::globe::ModelConfig &model_config,
                        const std::vector<bool> &model_flags,
                        const specfem::attenuation::Setup &attenuation_setup);

} // namespace specfem::io::mesh::impl::fortran::dim3_globe
