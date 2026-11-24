#pragma once

#include "specfem/assembly/assembly.hpp"
#include "specfem/enums.hpp"
#include "specfem/injection/session/points.hpp"
#include "specfem/injection/session/provider_session.hpp"
#include <functional>
#include <optional>
#include <string_view>
#include <vector>

namespace specfem {
namespace injection {

/**
 * @brief Quantities an injection method requests for points of one material
 * system.
 *
 * Returns the quantities, with the number of buffer components each occupies,
 * or @c std::nullopt if the method does not support the system.
 */
using QuantityPolicy =
    std::function<std::optional<std::vector<QuantityRequest>>(
        specfem::element::medium_tag, specfem::element::property_tag)>;

/**
 * @brief Build the points handed to an injection plugin from the assembled
 * injection surface.
 *
 * Every material system of the solver becomes a point group, in the same order
 * on every rank, including systems without points. Coordinates and normals are
 * shared with @c assembly.injection_surface, not copied. Each group carries its
 * material properties by name and, if any rank has points in it, the
 * quantities the method requests there.
 *
 * Collective: every rank must call it.
 *
 * @param assembly Assembly holding the injection surface and properties
 * @param quantities Quantities the injection method requests per system
 * @param method_name Name of the injection method, for error messages
 * @return Points for specfem::injection::ProviderSession::setup
 * @throws std::runtime_error on every rank if some rank has points in a system
 * the method does not support
 */
[[nodiscard]] InjectionPoints make_injection_points(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly,
    const QuantityPolicy &quantities, std::string_view method_name);

/**
 * @brief Simulation parameters handed to an injection plugin.
 *
 * @param assembly Assembly holding the time stepping and the global domain
 * bounds
 * @param nstep Total number of time steps
 * @return t0, dt, nstep and the global bounding box of the mesh
 */
[[nodiscard]] SimulationParameters make_simulation_parameters(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly,
    int nstep);

} // namespace injection
} // namespace specfem
