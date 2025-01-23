#include "IO/seismogram/writer.hpp"
#include "compute/interface.hpp"
#include <fstream>

void specfem::IO::seismogram_writer::write(
    specfem::compute::assembly &assembly) {
  auto &receivers = assembly.receivers;

  receivers.sync_seismograms();

  const auto seismogram_types = receivers.get_seismogram_types();

  for (auto [station_name, network_name, seismogram_type] :
       receivers.get_stations()) {

    std::vector<std::string> filename;
    switch (seismogram_type) {
    case specfem::wavefield::type::displacement:
      filename = { this->output_folder + "/" + network_name + "." +
                       station_name + ".S2.BXX.semd",
                   this->output_folder + "/" + network_name + "." +
                       station_name + ".S2.BXZ.semd" };
      break;
    case specfem::wavefield::type::velocity:
      filename = { this->output_folder + "/" + network_name + "." +
                       station_name + ".S2.BXX.semv",
                   this->output_folder + "/" + network_name + "." +
                       station_name + ".S2.BXZ.semv" };
      break;
    case specfem::wavefield::type::acceleration:
      filename = { this->output_folder + "/" + network_name + "." +
                       station_name + ".S2.BXX.sema",
                   this->output_folder + "/" + network_name + "." +
                       station_name + ".S2.BXZ.sema" };
      break;
    case specfem::wavefield::type::pressure:
      filename = { this->output_folder + "/" + network_name + "." +
                   station_name + ".S2.PRE.semp" };
      break;
    }

    const int ncomponents = filename.size();
    std::vector<std::ofstream> seismo_file(ncomponents);
    for (int icomp = 0; icomp < ncomponents; icomp++) {
      seismo_file[icomp].open(filename[icomp]);
    }

    for (auto [time, value] : receivers.get_seismogram(
             station_name, network_name, seismogram_type)) {
      for (int icomp = 0; icomp < ncomponents; icomp++) {
        seismo_file[icomp] << std::scientific << time << " " << std::scientific
                           << value[icomp] << "\n";
      }
    }

    for (int icomp = 0; icomp < ncomponents; icomp++) {
      seismo_file[icomp].close();
    }
  }
}
