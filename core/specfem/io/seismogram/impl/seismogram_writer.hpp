#pragma once

#include "specfem/enums.hpp"
#include <string>
#include <utility>
#include <vector>

namespace specfem {
namespace io {
namespace impl {

/**
 * @brief Format-dispatching helper for writing seismogram time series.
 *
 * The primary template is intentionally undefined. Include the corresponding
 * format header (e.g. ascii.hpp) to obtain the specialization for the desired
 * format, then call write_seismogram<Format>().
 */
template <specfem::enums::seismogram_format Format>
struct SeismogramFormatWriter;

/**
 * @brief Write seismogram time series in the given format.
 *
 * Dispatches to SeismogramFormatWriter<Format>::write(). To support a new
 * format, add a specialization of SeismogramFormatWriter in a dedicated header
 * and include it before calling this function.
 *
 * @tparam Format         Output format (e.g.
 * specfem::enums::seismogram_format::ascii).
 * @tparam SeismogramView Iterable of (time, value[]) pairs.
 * @param filenames       One output filename per component.
 * @param seismogram_view Range returned by receivers.get_seismogram().
 */
template <specfem::enums::seismogram_format Format, typename SeismogramView>
void write_seismogram(const std::vector<std::string> &filenames,
                      SeismogramView &&seismogram_view) {
  SeismogramFormatWriter<Format>::write(
      filenames, std::forward<SeismogramView>(seismogram_view));
}

} // namespace impl
} // namespace io
} // namespace specfem
