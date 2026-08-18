#pragma once

#include "seismogram_writer.hpp"
#include "specfem/assembly/receivers/impl/receiver_iterator.hpp"
#include "specfem/datetime.hpp"
#include "specfem/logger.hpp"
#include "specfem/mpi.hpp"
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace specfem {
namespace io {
namespace impl {

/**
 * @brief ASCII specialization of SeismogramFormatWriter.
 *
 * Iterates over all stations and seismogram types, writing one file per
 * component with (time, value) pairs in scientific notation.
 */
template <>
struct SeismogramFormatWriter<specfem::enums::seismogram_format::ascii> {
  template <typename Stations, typename Receivers>
  static void
  write(Stations &&stations, Receivers &receivers, ChannelGenerator &gen,
        const std::string &output_folder,
        std::optional<specfem::datetime::type> /*starttime*/ = std::nullopt) {
    const int my_rank = specfem::MPI::get_rank();

    specfem::Logger::debug("Writing ASCII seismograms to " + output_folder +
                           " from rank " + std::to_string(my_rank) + " (" +
                           std::to_string(stations.size()) + " stations)");

    const int nsteps = receivers.get_nsteps();

    for (auto station_info : stations) {
      for (auto seismogram_type : station_info.get_seismogram_types()) {
        const std::vector<std::string> filenames =
            gen.get_station_filenames<Receivers::dimension_tag>(
                station_info, seismogram_type);

        const int ncomponents = static_cast<int>(filenames.size());

        if (ncomponents == 0 || nsteps == 0)
          continue;

        std::vector<type_real> times(nsteps);
        std::vector<std::vector<type_real>> values(
            ncomponents, std::vector<type_real>(nsteps));

        int step = 0;
        for (auto [time, value] : receivers.get_seismogram(
                 station_info.station_name, station_info.network_name,
                 seismogram_type)) {
          times[step] = time;
          for (int icomp = 0; icomp < ncomponents; ++icomp)
            values[icomp][step] = value[icomp];
          ++step;
        }

        std::vector<std::ofstream> seismo_file(ncomponents);
        for (int icomp = 0; icomp < ncomponents; ++icomp) {
          seismo_file[icomp].open(output_folder + "/" + filenames[icomp]);
          if (!seismo_file[icomp].is_open()) {
            throw std::runtime_error("Could not open seismogram file: " +
                                     filenames[icomp]);
          }
        }

        for (int step = 0; step < nsteps; ++step) {
          for (int icomp = 0; icomp < ncomponents; ++icomp) {
            seismo_file[icomp] << std::scientific << times[step] << " "
                               << std::scientific << values[icomp][step]
                               << "\n";
          }
        }

        for (int icomp = 0; icomp < ncomponents; ++icomp) {
          seismo_file[icomp].close();

          specfem::Logger::debug(
              "ASCII file for station " + station_info.network_name + "." +
              station_info.station_name + " written to: " + filenames[icomp] +
              " from rank " + std::to_string(my_rank));
        }
      }
    }
  }
};

} // namespace impl
} // namespace io
} // namespace specfem
