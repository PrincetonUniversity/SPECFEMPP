#include "io/seismogram/writer.hpp"
#include "enumerations/specfem_enums.hpp"
#include "specfem/compute.hpp"
#include <fstream>

void specfem::io::seismogram_writer::write(
    specfem::compute::assembly &assembly) {
  auto &receivers = assembly.receivers;

  receivers.sync_seismograms();

  for (auto station_info : receivers.stations()) {
    std::string network_name = station_info.network_name;
    std::string station_name = station_info.station_name;

    for (auto seismogram_type : station_info.get_seismogram_types()) {

      std::vector<std::string> filename;
      switch (seismogram_type) {
      case specfem::wavefield::type::displacement:
        if (this->elastic_wave == specfem::enums::elastic_wave::sh) {
          filename = { this->output_folder + "/" + network_name + "." +
                       station_name + ".S2.BXY.semd" };
        } else if (this->elastic_wave == specfem::enums::elastic_wave::psv) {
          filename = { this->output_folder + "/" + network_name + "." +
                           station_name + ".S2.BXX.semd",
                       this->output_folder + "/" + network_name + "." +
                           station_name + ".S2.BXZ.semd" };
        }
        break;
      case specfem::wavefield::type::velocity:
        if (this->elastic_wave == specfem::enums::elastic_wave::sh) {
          filename = { this->output_folder + "/" + network_name + "." +
                       station_name + ".S2.BXY.semv" };
        } else if (this->elastic_wave == specfem::enums::elastic_wave::psv) {
          filename = { this->output_folder + "/" + network_name + "." +
                           station_name + ".S2.BXX.semv",
                       this->output_folder + "/" + network_name + "." +
                           station_name + ".S2.BXZ.semv" };
        }
        break;
      case specfem::wavefield::type::acceleration:
        if (this->elastic_wave == specfem::enums::elastic_wave::sh) {
          filename = { this->output_folder + "/" + network_name + "." +
                       station_name + ".S2.BXY.sema" };
        } else if (this->elastic_wave == specfem::enums::elastic_wave::psv) {
          filename = { this->output_folder + "/" + network_name + "." +
                           station_name + ".S2.BXX.sema",
                       this->output_folder + "/" + network_name + "." +
                           station_name + ".S2.BXZ.sema" };
        }
        break;
      case specfem::wavefield::type::pressure:
        if (this->elastic_wave == specfem::enums::elastic_wave::sh) {
          throw std::runtime_error(
              "Pressure seismograms are not supported for SH waves");
        } else if (this->elastic_wave == specfem::enums::elastic_wave::psv) {
          filename = { this->output_folder + "/" + network_name + "." +
                       station_name + ".S2.PRE.semp" };
        }
        break;
      case specfem::wavefield::type::rotation:
        if (this->elastic_wave == specfem::enums::elastic_wave::psv) {
          filename = { this->output_folder + "/" + network_name + "." +
                       station_name + ".S2.BXT.semr" };
        } else if (this->elastic_wave == specfem::enums::elastic_wave::sh) {
          // NEEDS TO BE UPDATED WHEN IMPLEMENTING SH_LV
          // L should be rotation around x and v rotation around z
          filename = { this->output_folder + "/" + network_name + "." +
                           station_name + ".S2.BXV.semr",
                       this->output_folder + "/" + network_name + "." +
                           station_name + ".S2.BXL.semr" };
          throw std::runtime_error(
              "Rotation seismograms are not supported for SH waves");
        }
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
