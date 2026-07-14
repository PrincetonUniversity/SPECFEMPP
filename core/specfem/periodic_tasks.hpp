#pragma once

#include "specfem/periodic_tasks/check_signal.hpp"
#include "specfem/periodic_tasks/checkpointing.hpp"
#include "specfem/periodic_tasks/periodic_task.hpp"
#include "specfem/periodic_tasks/plot_wavefield.hpp"
#include "specfem/periodic_tasks/wavefield_reader.hpp"
#include "specfem/periodic_tasks/wavefield_writer.hpp"

/**
 * @namespace specfem::periodic_tasks
 *
 * @brief Namespace for tasks that need to be performed periodically during the
 * simulation. This includes tasks such as plotting wavefields, checking for
 * user interrupts, and reading/writing wavefield data. These tasks are
 * implemented as subclasses of the `PeriodicTask` base class.
 */
namespace specfem::periodic_tasks {}
