#include "../include/writer.h"
#include "../include/compute.h"
#include "../include/receiver.h"
#include <fstream>

void specfem::writer::seismogram::write() {

  const int n_receivers = this->receivers.size();
  const int nsig_types = this->compute_receivers->h_seismogram_types.extent(0);
  const int nsig_steps = this->compute_receivers->h_seismogram.extent(0);
  const auto h_seismogram = this->compute_receivers->h_seismogram;
  const type_real dt = this->dt;
  const type_real t0 = this->t0;
  const type_real nstep_between_samples = this->nstep_between_samples;

  this->compute_receivers->sync_seismograms();

  std::cout << "output folder : " << this->output_folder << "\n";

  switch (this->type) {
  case specfem::seismogram::format::ascii:
    // Open stream
    for (int irec = 0; irec < n_receivers; irec++) {
      std::string network_name = receivers[irec]->get_network_name();
      std::string station_name = receivers[irec]->get_station_name();
      for (int isig = 0; isig < nsig_types; isig++) {
        std::vector<std::string> filename;
        auto stype = this->compute_receivers->h_seismogram_types(isig);
        switch (stype) {
        case specfem::seismogram::displacement:
          filename = { this->output_folder + "/" + network_name + station_name +
                           "BXX" + ".semd",
                       this->output_folder + "/" + network_name + station_name +
                           "BXZ" + ".semd" };
          break;
        case specfem::seismogram::velocity:
          filename = { this->output_folder + "/" + network_name + station_name +
                           "BXX" + ".semv",
                       this->output_folder + "/" + network_name + station_name +
                           "BXZ" + ".semv" };
          break;
        case specfem::seismogram::acceleration:
          filename = { this->output_folder + "/" + network_name + station_name +
                           "BXX" + ".sema",
                       this->output_folder + "/" + network_name + station_name +
                           "BXZ" + ".sema" };
          break;
        default:
          std::ostringstream message;
          message << "seismogram type " << stype
                  << " has not been implemented yet.";
          throw std::runtime_error(message.str());
        }

        for (int iorientation = 0; iorientation < filename.size();
             iorientation++) {
          std::ofstream seismo_file;
          seismo_file.open(filename[iorientation]);
          for (int isig_step = 0; isig_step < nsig_steps; isig_step++) {
            const type_real time_t =
                isig_step * dt * nstep_between_samples + t0;
            const type_real value =
                h_seismogram(isig_step, isig, irec, iorientation);

            seismo_file << std::scientific << time_t << " " << std::scientific
                        << value << "\n";
          }
          seismo_file.close();
        }
      }
    }
    break;
  default:
    std::ostringstream message;
    message << "seismogram output type " << this->type
            << " has not been implemented yet.";
    throw std::runtime_error(message.str());
  }

  std::cout << std::endl;
}
