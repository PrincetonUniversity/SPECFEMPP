#include "compute/interface.hpp"
#include "writer/interface.hpp"
#include <fstream>

void specfem::writer::seismogram::write() {

  this->receivers.sync_seismograms();

  const auto seismogram_types = this->receivers.get_seismogram_types();

  for (auto [station_name, network_name, seismogram_type] :
       this->receivers.get_stations()) {

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

    for (auto [time, value] : this->receivers.get_seismogram(
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

// void specfem::writer::seismogram::write() {

//   const int nsig_types = this->receivers.h_seismogram_types.extent(0);
//   const int nsig_steps = this->receivers.h_seismogram.extent(0);
//   const auto h_seismogram = this->receivers.h_seismogram;
//   const type_real dt = this->dt;
//   const type_real t0 = this->t0;
//   const type_real nstep_between_samples = this->nstep_between_samples;

//   this->receivers.sync_seismograms();

//   std::cout << "output folder : " << this->output_folder << "\n";

//   switch (this->type) {
//   case specfem::enums::seismogram::ascii:
//     // Open stream
//     for (int irec = 0; irec < nreceivers; irec++) {
//       std::string network_name = receivers.network_names[irec];
//       std::string station_name = receivers.station_names[irec];
//       for (int isig = 0; isig < nsig_types; isig++) {
//         std::vector<std::string> filename;
//         auto stype = this->receivers.h_seismogram_types(isig);
//         switch (stype) {
//         case specfem::enums::seismogram::type::displacement:
//           filename = { this->output_folder + "/" + network_name +
//           station_name +
//                            "BXX" + ".semd",
//                        this->output_folder + "/" + network_name +
//                        station_name +
//                            "BXZ" + ".semd" };
//           break;
//         case specfem::enums::seismogram::type::velocity:
//           filename = { this->output_folder + "/" + network_name +
//           station_name +
//                            "BXX" + ".semv",
//                        this->output_folder + "/" + network_name +
//                        station_name +
//                            "BXZ" + ".semv" };
//           break;
//         case specfem::enums::seismogram::type::acceleration:
//           filename = { this->output_folder + "/" + network_name +
//           station_name +
//                            "BXX" + ".sema",
//                        this->output_folder + "/" + network_name +
//                        station_name +
//                            "BXZ" + ".sema" };
//           break;
//         case specfem::enums::seismogram::type::pressure:
//           filename = { this->output_folder + "/" + network_name +
//           station_name +
//                        "PRE" + ".semp" };
//           break;
//         }

//         for (int iorientation = 0; iorientation < filename.size();
//              iorientation++) {
//           std::ofstream seismo_file;
//           seismo_file.open(filename[iorientation]);
//           for (int isig_step = 0; isig_step < nsig_steps; isig_step++) {
//             const type_real time_t =
//                 isig_step * dt * nstep_between_samples + t0;
//             const type_real value =
//                 h_seismogram(isig_step, isig, irec, iorientation);

//             seismo_file << std::scientific << time_t << " " <<
//             std::scientific
//                         << value << "\n";
//           }
//           seismo_file.close();
//         }
//       }
//     }
//     break;
//   default:
//     std::ostringstream message;
//     message << "seismogram output type " << this->type
//             << " has not been implemented yet.";
//     throw std::runtime_error(message.str());
//   }

//   std::cout << std::endl;
// }
