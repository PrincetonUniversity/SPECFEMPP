#pragma once

#include "impl/ascii.hpp"
#include "impl/channel_generator.hpp"
#include "impl/sac.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/constants.hpp"
#include "specfem/enums.hpp"
#include "specfem/io/writer.hpp"
#include "specfem/setup.hpp"
#include <optional>
#include <stdexcept>
#include <vector>

namespace specfem {
namespace io {
/**
 * @brief Writer for outputting seismograms at receiver locations
 *
 * Records displacement, velocity, or acceleration at receiver stations and
 * writes time series to disk. Supports multiple output formats and wave types,
 * with configurable sampling intervals.
 */
class seismogram_writer : public writer, public impl::ChannelGenerator {

public:
  /**
   * @brief Construct a new seismogram writer object
   *
   * @param type Format of the output file
   * @param output_folder path to output folder where results will be stored
   * @param dt Time interval between subsequent timesteps
   * @param t0 Solver start time
   * @param nstep_between_samples number of timesteps between seismogram
   * sampling (seismogram sampling frequency)
   */
  seismogram_writer(
      const specfem::enums::seismogram_format type,
      const specfem::enums::elastic_wave elastic_wave,
      const specfem::enums::electromagnetic_wave electromagnetic_wave,
      const std::string output_folder, const type_real dt,
      const int nstep_between_samples,
      const std::optional<bool> write_from_main = std::nullopt)
      : impl::ChannelGenerator(dt, nstep_between_samples), type(type),
        elastic_wave(elastic_wave), electromagnetic_wave(electromagnetic_wave),
        output_folder(output_folder), dt(dt),
        nstep_between_samples(nstep_between_samples),
        write_from_main(write_from_main) {};

  void write(specfem::assembly::assembly<specfem::element::dimension_tag::dim2>
                 &assembly) override {
    write_impl(assembly);
  }

  void write(specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
                 &assembly) override {
    write_impl(assembly);
  }

private:
  template <specfem::element::dimension_tag DimensionTag>
  void write_impl(specfem::assembly::assembly<DimensionTag> &assembly) {
    auto &receivers = assembly.receivers;
    receivers.sync_seismograms();

    const int my_rank = specfem::MPI::get_rank();
    const bool is_main = specfem::MPI::main_proc();
    auto rank_filter =
        [&](const specfem::assembly::receivers_impl::StationInfo &s) {
          return s.partition_index == my_rank || is_main;
        };
    auto filtered_stations = receivers.stations(rank_filter);

    switch (type) {
    case specfem::enums::seismogram_format::ascii:
      specfem::io::impl::write_seismogram<
          specfem::enums::seismogram_format::ascii>(
          filtered_stations, receivers, *this, output_folder);
      break;
    case specfem::enums::seismogram_format::sac:
      specfem::io::impl::write_seismogram<
          specfem::enums::seismogram_format::sac>(filtered_stations, receivers,
                                                  *this, output_folder);
      break;
    default:
      throw std::runtime_error(
          "Unsupported seismogram output format requested.");
    }
  }

  specfem::enums::seismogram_format type; ///< Output format of the seismogram
                                          ///< file
  std::string output_folder; ///< Path to output folder where results will be
                             ///< stored
  type_real dt;              ///< Time interval between subsequent timesteps
  specfem::enums::elastic_wave elastic_wave; ///< Type of wavefield (Writes
                                             ///< .BXY for SH waves and .BXX,
                                             ///< .BXZ for P-SV waves)
  specfem::enums::electromagnetic_wave
      electromagnetic_wave;  ///< Type of
                             ///< electromagnetic
                             ///< wavefield
  int nstep_between_samples; ///< number of timesteps between seismogram
                             ///< sampling (seismogram sampling frequency)
  std::optional<bool> write_from_main; ///< Gather all data to rank 0 and write
                                       ///< from main proc
};

} // namespace io
} // namespace specfem
