#pragma once

/**
 * @brief Data structures for medium definition and property storage.
 *
 * Provides containers for defining material parameters and storing
 * precomputed domain properties and kernels required for the simulation.
 *
 * **Core components:**
 * - `material<>`: Definition of physical material parameters (density,
 * velocity, etc.)
 * - `domain_properties<>`: Storage for element-wise or point-wise material
 * properties
 * - `domain_kernels<>`: Storage for precomputed kernels used in assembly
 */
namespace specfem::medium_container {}

#include "specfem/medium_container/domain_kernels.hpp"
#include "specfem/medium_container/domain_properties.hpp"
#include "specfem/medium_container/material.hpp"
#include "specfem/medium_container/point_kernels.hpp"
#include "specfem/medium_container/point_properties.hpp"
