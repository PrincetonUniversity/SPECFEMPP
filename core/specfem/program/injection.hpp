#pragma once

#include "specfem/injection.hpp"

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

} // namespace program_impl
} // namespace specfem
