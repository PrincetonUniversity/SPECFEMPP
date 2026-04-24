#pragma once

#include "seismogram_writer.hpp"
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace specfem {
namespace io {
namespace impl {

/**
 * @brief ASCII specialization of SeismogramFormatWriter: writes one file per
 *        component with (time, value) pairs in scientific notation.
 */
template <>
struct SeismogramFormatWriter<specfem::enums::seismogram_format::ascii> {
  template <typename SeismogramView>
  static void write(const std::vector<std::string> &filenames,
                    SeismogramView &&seismogram_view) {
    const int ncomponents = static_cast<int>(filenames.size());
    std::vector<std::ofstream> seismo_file(ncomponents);
    for (int icomp = 0; icomp < ncomponents; icomp++) {
      seismo_file[icomp].open(filenames[icomp]);
      if (!seismo_file[icomp].is_open()) {
        throw std::runtime_error("Could not open seismogram file: " +
                                 filenames[icomp]);
      }
    }

    for (auto [time, value] : seismogram_view) {
      for (int icomp = 0; icomp < ncomponents; icomp++) {
        seismo_file[icomp] << std::scientific << time << " " << std::scientific
                           << value[icomp] << "\n";
      }
    }

    for (int icomp = 0; icomp < ncomponents; icomp++) {
      seismo_file[icomp].close();
    }
  }
};

} // namespace impl
} // namespace io
} // namespace specfem
