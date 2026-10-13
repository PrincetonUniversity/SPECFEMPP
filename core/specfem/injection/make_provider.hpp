#pragma once

#include "specfem/element/tags.hpp"
#include "specfem/injection/injection_inputs.hpp"
#include "specfem/setup.hpp"
#include <memory>

namespace specfem {
namespace injection {

template <specfem::element::dimension_tag DimensionTag>
class injection_provider;
struct injection_points;

/**
 * @brief Construct (but do not initialize) an injection provider.
 *
 * Single source of truth for method dispatch. The method-agnostic @p points
 * are forwarded to the method, which owns all field/traction derivation. The
 * caller runs @c initialize() afterwards.
 *
 * @tparam DimensionTag Spatial dimension (@c dim2 or @c dim3).
 * @param inputs Injection domain inputs (method, model, wave, window params).
 * @param points Method-agnostic injection points (geometry, medium, material).
 * @param dt     Simulation time step in seconds.
 * @param nstep  Total number of simulation time steps.
 * @return Shared pointer to the constructed provider.
 * @throws std::runtime_error if @c inputs.method is not supported.
 */
template <specfem::element::dimension_tag DimensionTag>
std::shared_ptr<injection_provider<DimensionTag>>
make_provider(const injection_inputs &inputs, const injection_points &points,
              type_real dt, int nstep);

} // namespace injection
} // namespace specfem
