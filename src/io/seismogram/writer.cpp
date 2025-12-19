#include "io/seismogram/writer.hpp"
#include "enumerations/interface.hpp"
#include "enumerations/specfem_enums.hpp"
#include "specfem/assembly.hpp"
#include "utilities/utilities.hpp"
#include <fstream>

void specfem::io::seismogram_writer::write(
    specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly) {
  auto &receivers = assembly.receivers;

  receivers.sync_seismograms();

  for (auto station_info : receivers.stations()) {
    std::string network_name = station_info.network_name;
    std::string station_name = station_info.station_name;

    for (auto seismogram_type : station_info.get_seismogram_types()) {

      std::vector<std::string> filenames;
      switch (seismogram_type) {
      case specfem::wavefield::type::displacement:
        if (this->elastic_wave == specfem::enums::elastic_wave::sh) {
          filenames = { this->output_folder + "/" + network_name + "." +
                        station_name + ".S2.BXY.semd" };
        } else if (this->elastic_wave == specfem::enums::elastic_wave::psv) {
          filenames = { this->output_folder + "/" + network_name + "." +
                            station_name + ".S2.BXX.semd",
                        this->output_folder + "/" + network_name + "." +
                            station_name + ".S2.BXZ.semd" };
        }
        break;
      case specfem::wavefield::type::velocity:
        if (this->elastic_wave == specfem::enums::elastic_wave::sh) {
          filenames = { this->output_folder + "/" + network_name + "." +
                        station_name + ".S2.BXY.semv" };
        } else if (this->elastic_wave == specfem::enums::elastic_wave::psv) {
          filenames = { this->output_folder + "/" + network_name + "." +
                            station_name + ".S2.BXX.semv",
                        this->output_folder + "/" + network_name + "." +
                            station_name + ".S2.BXZ.semv" };
        }
        break;
      case specfem::wavefield::type::acceleration:
        if (this->elastic_wave == specfem::enums::elastic_wave::sh) {
          filenames = { this->output_folder + "/" + network_name + "." +
                        station_name + ".S2.BXY.sema" };
        } else if (this->elastic_wave == specfem::enums::elastic_wave::psv) {
          filenames = { this->output_folder + "/" + network_name + "." +
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
          filenames = { this->output_folder + "/" + network_name + "." +
                        station_name + ".S2.PRE.semp" };
        }
        break;
      // There is no naming convention for rotation so we use [B]road [X]
      // computer generated [Y] rotation axis for `P_SV_T` and extension `.semr`
      // for spectral element rotation
      case specfem::wavefield::type::rotation:
        if (this->elastic_wave == specfem::enums::elastic_wave::psv) {
          filenames = { this->output_folder + "/" + network_name + "." +
                        station_name + ".S2.BXY.semr" };
        } else if (this->elastic_wave == specfem::enums::elastic_wave::sh) {
          // NEEDS TO BE UPDATED WHEN IMPLEMENTING SH_LV
          // L should be rotation around x and v rotation around z
          filenames = { this->output_folder + "/" + network_name + "." +
                            station_name + ".S2.BXX.semr",
                        this->output_folder + "/" + network_name + "." +
                            station_name + ".S2.BXZ.semr" };
          throw std::runtime_error(
              "Rotation seismograms are not supported for SH waves");
        }
        break;
        // There is no naming convention for intrinsic rotation so
      case specfem::wavefield::type::intrinsic_rotation:
        if (this->elastic_wave == specfem::enums::elastic_wave::psv) {
          filenames = { this->output_folder + "/" + network_name + "." +
                        station_name + ".S2.BXY.semir" };
        } else if (this->elastic_wave == specfem::enums::elastic_wave::sh) {
          // NEEDS TO BE UPDATED WHEN IMPLEMENTING SH_LV
          // L should be rotation around x and v rotation around z
          filenames = { this->output_folder + "/" + network_name + "." +
                            station_name + ".S2.BXX.semir",
                        this->output_folder + "/" + network_name + "." +
                            station_name + ".S2.BXZ.semir" };
          throw std::runtime_error(
              "Intrinsic rotation seismograms are not supported for SH waves");
        }
        break;
      case specfem::wavefield::type::curl:
        if (this->elastic_wave == specfem::enums::elastic_wave::psv) {
          filenames = { this->output_folder + "/" + network_name + "." +
                        station_name + ".S2.BXY.semc" };
        } else if (this->elastic_wave == specfem::enums::elastic_wave::sh) {
          filenames = { this->output_folder + "/" + network_name + "." +
                            station_name + ".S2.BXX.semc",
                        this->output_folder + "/" + network_name + "." +
                            station_name + ".S2.BXZ.semc" };
        }
        break;
      default:
        std::ostringstream message;
        message << "Error reading specfem receiver configuration. (" << __FILE__
                << ":" << __LINE__ << ")\n";
        message << "Unknown seismogram type: "
                << specfem::wavefield::to_string(seismogram_type) << "\n";
        message
            << "Valid seismogram types are: displacement, velocity, "
            << "acceleration, pressure, rotation, intrinsic_rotation, curl.\n";
        message << "Please check your configuration file.\n";
        throw std::runtime_error(message.str());
      }

      const int ncomponents = filenames.size();
      std::vector<std::ofstream> seismo_file(ncomponents);
      for (int icomp = 0; icomp < ncomponents; icomp++) {
        seismo_file[icomp].open(filenames[icomp]);
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

void specfem::io::seismogram_writer::write(
    specfem::assembly::assembly<specfem::dimension::type::dim3> &assembly) {

  // Get reference to receivers and timestep
  auto &receivers = assembly.receivers;
  auto dt = receivers.get_timestep();

  // Get channel code depending on the time range
  receivers.sync_seismograms();

  // Initialize filename generator

  // Loop over all stations
  for (auto station_info : receivers.stations()) {

    std::string network_name = station_info.network_name;
    std::string station_name = station_info.station_name;

    // Loop over all seismogram types for this station
    for (auto seismogram_type : station_info.get_seismogram_types()) {

      // Depending on station name and wavefield type, get the correct filenames
      std::vector<std::string> filenames = this->get_station_filenames(
          network_name, station_name, "S3", seismogram_type);

      const int ncomponents = filenames.size();
      std::vector<std::ofstream> seismo_file(ncomponents);
      for (int icomp = 0; icomp < ncomponents; icomp++) {
        seismo_file[icomp].open(filenames[icomp]);
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
