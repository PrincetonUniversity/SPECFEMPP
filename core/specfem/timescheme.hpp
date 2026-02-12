#pragma once

/**
 * @namespace specfem::time_scheme
 * @brief Time integration schemes for wave propagation
 *
 * Provides base classes and utilities for time-stepping algorithms used in
 * spectral element wave propagation simulations. Supports both forward and
 * backward time integration for regular and adjoint simulations.
 *
 * The `time_scheme` namespace defines the interface for time integration
 * schemes. The base class `time_scheme` provides methods for iterating over
 * time steps, managing seismogram output, and applying predictor-corrector
 * phases. Specific schemes (e.g., `newmark`) are implemented in derived
 * classes.
 */
namespace specfem::time_scheme {}

// Base class
#include "specfem/timescheme/timescheme.hpp"

// Specific time schemes: Newmark
#include "specfem/timescheme/newmark.hpp"
