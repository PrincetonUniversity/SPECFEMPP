#pragma once

#include "seismogram_writer.hpp"
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
  template <typename Receivers>
  static void write(Receivers &receivers, ChannelGenerator &gen,
                    const std::string &output_folder,
                    const std::optional<bool> write_from_main = std::nullopt) {
    const bool from_main = write_from_main.value_or(false);

    for (auto station_info : receivers.stations()) {
#ifdef SPECFEM_ENABLE_MPI
      if (station_info.islice != specfem::MPI::get_rank() &&
          !(from_main && specfem::MPI::main_proc())) {
        continue; // Skip stations not assigned to this rank
      }
#endif
      for (auto seismogram_type : station_info.get_seismogram_types()) {
        const std::vector<std::string> filenames =
            gen.get_station_filenames<Receivers::dimension_tag>(
                station_info, seismogram_type);

        const int ncomponents = static_cast<int>(filenames.size());
        std::vector<std::ofstream> seismo_file(ncomponents);
        for (int icomp = 0; icomp < ncomponents; icomp++) {
          seismo_file[icomp].open(output_folder + "/" + filenames[icomp]);
          if (!seismo_file[icomp].is_open()) {
            throw std::runtime_error("Could not open seismogram file: " +
                                     filenames[icomp]);
          }
        }

        for (auto [time, value] : receivers.get_seismogram(
                 station_info.station_name, station_info.network_name,
                 seismogram_type)) {
          for (int icomp = 0; icomp < ncomponents; icomp++) {
            seismo_file[icomp] << std::scientific << time << " "
                               << std::scientific << value[icomp] << "\n";
          }
        }

        for (int icomp = 0; icomp < ncomponents; icomp++) {
          seismo_file[icomp].close();
        }
      }
    }
  }
};

} // namespace impl
} // namespace io
} // namespace specfem
