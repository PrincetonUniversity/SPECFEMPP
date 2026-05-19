#pragma once

#include "channel_generator.hpp"
#include "specfem/enums.hpp"
#include <optional>
#include <string>
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
 * @brief Write seismograms for all stations in the given format.
 *
 * Delegates entirely to SeismogramFormatWriter<Format>::write().
 *
 * @tparam Format    Output format (e.g.
 * specfem::enums::seismogram_format::ascii).
 * @tparam Stations  Range of StationInfo (e.g. a StationView).
 * @tparam Receivers Type exposing get_seismogram() and dimension_tag.
 * @param stations   Filtered station range to write.
 * @param receivers  Receivers object after sync_seismograms() has been called.
 * @param gen        ChannelGenerator used to build output filenames.
 * @param send_to_main Whether to send seismograms to the main process.
 */
template <specfem::enums::seismogram_format Format, typename Stations,
          typename Receivers>
void write_seismogram(Stations &&stations, Receivers &receivers,
                      ChannelGenerator &gen, const std::string &output_folder,
                      const bool send_to_main) {
  SeismogramFormatWriter<Format>::write(std::forward<Stations>(stations),
                                        receivers, gen, output_folder,
                                        send_to_main);
}

} // namespace impl
} // namespace io
} // namespace specfem
