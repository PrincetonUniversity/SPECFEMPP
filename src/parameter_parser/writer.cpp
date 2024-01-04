#include "constants.hpp"
#include "parameter_parser/interface.hpp"
#include "writer/interface.hpp"
#include "yaml-cpp/yaml.h"
#include <memory>

std::shared_ptr<specfem::writer::writer>
specfem::runtime_configuration::seismogram::instantiate_seismogram_writer(
    std::vector<std::shared_ptr<specfem::receivers::receiver> > &receivers,
    specfem::compute::receivers &compute_receivers, const type_real dt,
    const type_real t0, const int nstep_between_samples) const {

  specfem::enums::seismogram::format type;
  if (this->seismogram_format == "seismic_unix" ||
      this->seismogram_format == "su") {
    type = specfem::enums::seismogram::format::seismic_unix;
  } else if (this->seismogram_format == "ascii") {
    type = specfem::enums::seismogram::format::ascii;
  }

  std::shared_ptr<specfem::writer::writer> writer =
      std::make_shared<specfem::writer::seismogram>(
          receivers, compute_receivers, type, this->output_folder, dt, t0,
          nstep_between_samples);

  return writer;
}
