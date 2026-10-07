#pragma once

/**
 * @brief Method-agnostic abstract base for injection field providers.
 *
 * Defines the host-side interface through which an injection method (FK,
 * AxiSEM3D, DSM, file-based, …) exposes a uniform device frame buffer to the
 * stiffness kernel.  Polymorphism lives entirely on the host: @c initialize
 * and @c ensure_window are called from host code; the kernel itself reads only
 * the method-agnostic @c InjectionFrameBuffer through @c load_on_device.
 *
 * To add a new injection method, derive from this class and implement the three
 * pure virtual functions.  The base class names no specific method.
 */

#include "specfem/element/tags.hpp"
#include "specfem/injection/injection_frame_buffer.hpp"

namespace specfem {
namespace injection {

/**
 * @brief Abstract base class for injection field providers.
 *
 * @tparam DimensionTag Spatial dimension of the target simulation
 *                      (@c specfem::element::dimension_tag::dim2 or @c dim3).
 *
 * Concrete subclasses (e.g. @c fk_provider) implement @c initialize and
 * @c ensure_window on the host, and expose the populated @c
 * InjectionFrameBuffer through @c frame_buffer().  The stiffness kernel is
 * blind to the concrete method: it only calls @c
 * frame_buffer().load_on_device(istep, point, comp).
 *
 * Future methods (AxiSEM3D coupling, DSM, file-based replay) are separate
 * subclasses with no changes to this interface.
 */
template <specfem::element::dimension_tag DimensionTag>
class injection_provider {
public:
  /** @brief Virtual destructor for safe polymorphic deletion. */
  virtual ~injection_provider() = default;

  /**
   * @brief Host preprocessing: compute or prepare the injected field.
   *
   * For FK: runs the wavenumber-frequency solve and stores the resulting
   * cubic-B-spline coefficient Views.  Must be called once before the first
   * @c ensure_window call.
   */
  virtual void initialize() = 0;

  /**
   * @brief Ensure the device frame buffer covers time step @p istep.
   *
   * For FK (first-cut, full-storage): reconstructs the full displacement and
   * traction time series from the stored B-spline coefficients on the first
   * call; subsequent calls are no-ops.  Future windowed providers may load
   * only the relevant time-chunk around @p istep.
   *
   * @param istep Simulation time step index being requested.
   */
  virtual void ensure_window(int istep) = 0;

  /**
   * @brief Device-resident uniform frame buffer the stiffness kernel reads.
   *
   * Call @c ensure_window before accessing the buffer for a given time step.
   *
   * @return const reference to the populated InjectionFrameBuffer
   */
  virtual const InjectionFrameBuffer &frame_buffer() const = 0;
};

} // namespace injection
} // namespace specfem
