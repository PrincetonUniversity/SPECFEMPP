#pragma once

#include "channel_generator.hpp"
#include "specfem/enums.hpp"
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
 * Delegates entirely to SeismogramFormatWriter<Format>::write(), which is
 * responsible for iterating over stations and writing each one. This allows
 * format implementations to decide how to group or batch station output
 * (e.g. one file per station for ASCII, or a single container for HDF5).
 *
 * @tparam Format    Output format (e.g.
 * specfem::enums::seismogram_format::ascii).
 * @tparam Receivers Type exposing stations(), get_seismogram(), and
 * dimension_tag.
 * @param receivers  Receivers object after sync_seismograms() has been called.
 * @param gen        ChannelGenerator used to build output filenames.
 */
template <specfem::enums::seismogram_format Format, typename Receivers>
void write_seismogram(Receivers &receivers, ChannelGenerator &gen,
                      const std::string &output_folder) {
  SeismogramFormatWriter<Format>::write(receivers, gen, output_folder);
}

} // namespace impl
} // namespace io
} // namespace specfem
