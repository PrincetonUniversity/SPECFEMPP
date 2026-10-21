#pragma once

#include "injection/options.hpp"
#include "injection/plugin_library.hpp"
#include "injection/request.hpp"

/**
 * @namespace specfem::injection
 * @brief Wavefield injection through externally provided plugins
 *
 * Loads injection plugins built against the plugin SDK
 * (specfem/plugin/injection_abi.h), verifies them before any of their code
 * runs, and resolves the command-line and configuration options that select
 * them.
 */
