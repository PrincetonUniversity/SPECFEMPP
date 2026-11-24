#pragma once

#include "specfem/assembly/assembly.hpp"
#include "specfem/injection.hpp"
#include "specfem/injection/method/registry.hpp"
#include "specfem/runtime_configuration/injection.hpp"
#include <memory>

namespace specfem {
namespace program_impl {

/**
 * @brief Load the injection plugin for a run, on every rank.
 *
 * Every rank opens the plugin, then the ranks agree on the outcome so that a
 * failure on any rank fails the run on all of them. Logs the loaded plugin
 * once per run.
 *
 * @param options Plugin selected for the run
 * @return The loaded plugin
 * @throws std::runtime_error on every rank if loading failed on any rank
 */
specfem::injection::PluginLibrary
load_injection_plugin(const specfem::injection::PluginOptions &options);

/**
 * @brief Build the injection method for a run and set it up with the plugin.
 *
 * Creates the method from @p registry, starts the plugin's provider session
 * (the plugin's @c create), hands the plugin the method's injection points
 * (its @c setup), and gives the session to the method. Collective: every rank
 * must call it.
 *
 * @param library Loaded plugin
 * @param options Plugin selected for the run (its configuration file is
 * passed to the plugin)
 * @param config The @c parameters.injection section
 * @param assembly Assembly holding the injection surface
 * @param nstep Total number of time steps
 * @param registry Injection methods available by name
 * @return The method, ready for the solver
 * @throws std::runtime_error on every rank if the mesh declares no injection
 * faces, or if creating or setting up the plugin fails on any rank
 */
std::shared_ptr<
    specfem::injection::InjectionMethod<specfem::element::dimension_tag::dim3>>
make_injection_method(
    specfem::injection::PluginLibrary library,
    const specfem::injection::PluginOptions &options,
    const specfem::runtime_configuration::Injection &config,
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly,
    int nstep, const specfem::injection::MethodRegistry &registry);

} // namespace program_impl
} // namespace specfem
