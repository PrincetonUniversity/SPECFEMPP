#pragma once

#include "specfem/assembly/assembly.hpp"
#include "specfem/element.hpp"
#include "specfem/simulation.hpp"
#include <string_view>

namespace specfem {
namespace injection {

/**
 * @brief The part of an injection method the time-marching solver sees.
 *
 * Each time step, the solver calls prepare_step() once, then apply() for every
 * medium right after the source interaction, before the stiffness kernels,
 * the MPI exchange and the mass-matrix division. Step @c istep holds the
 * incident field at simulation time @f$ t_0 + \text{istep} \, \Delta t @f$.
 *
 * Methods are built and set up by the program through
 * specfem::injection::InjectionMethod; this header keeps the solver free of the
 * plugin and session types.
 *
 * @tparam DimensionTag Spatial dimension
 */
template <specfem::element::dimension_tag DimensionTag> class CouplingMethod {
public:
  virtual ~CouplingMethod() = default;

  /**
   * @brief Name of the method, as selected in the parameter file.
   * @return Method name
   */
  [[nodiscard]] virtual std::string_view name() const = 0;

  /**
   * @brief Make the incident field for a step available. Collective.
   * @param istep Time step about to be computed
   */
  virtual void prepare_step(int istep) = 0;

  /**
   * @brief Add the injected term for one medium to the field's acceleration.
   *
   * @param assembly Assembly being time-stepped
   * @param medium Medium being updated
   * @param wavefield Wavefield being updated
   * @param istep Current time step
   */
  virtual void apply(specfem::assembly::assembly<DimensionTag> &assembly,
                     specfem::element::medium_tag medium,
                     specfem::simulation::field_type wavefield, int istep) = 0;
};

} // namespace injection
} // namespace specfem
