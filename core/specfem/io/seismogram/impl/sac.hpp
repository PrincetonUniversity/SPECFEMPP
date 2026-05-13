#pragma once

#include "seismogram_writer.hpp"
#include "specfem/element/tags.hpp"
#include "specfem/enums.hpp"
#include "specfem/enums/wavefield.hpp"
#include "specfem/mpi.hpp"
#include "specfem/setup.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace specfem {
namespace io {
namespace impl {

// ---------------------------------------------------------------------------
// SAC binary format helpers
// ---------------------------------------------------------------------------

static constexpr float UNDEF_F = -12345.0f;
static constexpr int32_t UNDEF_I = -12345;

// ---------------------------------------------------------------------------
// Component orientation: (CMPAZ [deg], CMPINC [deg]) for X/Y/Z indices.
// Convention: CMPAZ = azimuth from North clockwise; CMPINC = 0 → vertical up,
// 90 → horizontal.  Matches the Fortran write_output_SAC.f90 convention.
// ---------------------------------------------------------------------------
struct Orientation {
  float cmpaz, cmpinc;
};

constexpr std::array<Orientation, 3> kComponentOrientations = { {
    { 90.0f, 90.0f }, // X – East,  horizontal
    { 0.0f, 90.0f },  // Y – North, horizontal
    { 0.0f, 0.0f },   // Z – Up,    vertical
} };

constexpr Orientation kOrientationUndef = { UNDEF_F, UNDEF_F };

// ---------------------------------------------------------------------------
// SAC IDEP code per wavefield type.
// (6 = displacement/nm, 7 = velocity/nm·s⁻¹, 8 = acceleration/nm·s⁻²,
//  5 = IUNKN for pressure which has no standard SAC amplitude type)
// Using a function rather than an array to remain correct if the enum order
// ever changes.
// ---------------------------------------------------------------------------
inline int32_t get_idep_for_wavefield(specfem::enums::wavefield type) {
  switch (type) {
  case specfem::enums::wavefield::displacement:
    return 6; // IDISP
  case specfem::enums::wavefield::velocity:
    return 7; // IVEL
  case specfem::enums::wavefield::acceleration:
    return 8; // IACC
  case specfem::enums::wavefield::pressure:
    return 5; // IUNKN – pressure has no standard SAC amplitude type
  default:
    return UNDEF_I;
  }
}

// ---------------------------------------------------------------------------
// Extract the SEED channel code from a base filename of the form
// network.station.location.channel.extension (returns empty string on parse
// failure).
// ---------------------------------------------------------------------------
inline std::string
extract_channel_code_from_filename(const std::string &filename) {
  const auto ext_dot = filename.rfind('.');
  if (ext_dot == std::string::npos || ext_dot == 0)
    return "";
  const auto chan_dot = filename.rfind('.', ext_dot - 1);
  if (chan_dot == std::string::npos)
    return "";
  return filename.substr(chan_dot + 1, ext_dot - chan_dot - 1);
}

// ---------------------------------------------------------------------------
// Map the orientation letter (last character of the channel code) to a SAC
// (CMPAZ, CMPINC) orientation.
// ---------------------------------------------------------------------------
inline Orientation orientation_for_channel(const std::string &channel_code) {
  if (channel_code.empty())
    return kOrientationUndef;
  switch (channel_code.back()) {
  case 'X':
    return kComponentOrientations[0];
  case 'Y':
    return kComponentOrientations[1];
  case 'Z':
    return kComponentOrientations[2];
  default:
    return kOrientationUndef;
  }
}

// ---------------------------------------------------------------------------
// Write a left-justified, space-padded string into a fixed-width SAC text slot.
// ---------------------------------------------------------------------------
inline void fill_text_field(char *dst, std::string_view sv, int len) noexcept {
  std::fill_n(dst, len, ' ');
  std::memcpy(dst, sv.data(), std::min(static_cast<int>(sv.size()), len));
}

// ---------------------------------------------------------------------------
/**
 * @brief Write a single-component SAC binary (NVHDR=6) file.
 *
 * Binary layout (632-byte header followed by NPTS × float32 data):
 *   • 70 × float32  (280 bytes) – real-valued header words
 *   • 40 × int32   (160 bytes) – integer header words
 *   • 192 bytes                – text header words
 *
 * Field layout follows
 * https://ds.iris.edu/files/sac-manual/manual/file_format.html
 *
 * @param filepath      Output file path.
 * @param station_name  SEED station code ≤8 chars → KSTNM.
 * @param network_name  SEED network code ≤8 chars → KNETWK.
 * @param location_code SEED location ID  ≤8 chars → KHOLE.
 * @param channel_code  SEED channel code ≤8 chars → KCMPNM.
 * @param delta         Sample interval [s] → DELTA.
 * @param b             Begin time relative to reference [s] → B.
 * @param orientation   Component (CMPAZ, CMPINC) in degrees.
 * @param idep          SAC dependent-variable type code → IDEP.
 * @param samples       Single-component seismogram samples.
 */
// ---------------------------------------------------------------------------
inline void write_sac_binary(const std::string &filepath,
                             std::string_view station_name,
                             std::string_view network_name,
                             std::string_view location_code,
                             std::string_view channel_code, float delta,
                             float b, Orientation orientation, int32_t idep,
                             const std::vector<float> &samples) {
  const int32_t npts = static_cast<int32_t>(samples.size());

  // ── Compute statistics ───────────────────────────────────────────────────
  float depmin = samples.empty() ? UNDEF_F : samples[0];
  float depmax = samples.empty() ? UNDEF_F : samples[0];
  double sum = 0.0;
  for (float v : samples) {
    if (v < depmin)
      depmin = v;
    if (v > depmax)
      depmax = v;
    sum += v;
  }
  float depmen = samples.empty() ? UNDEF_F : static_cast<float>(sum / npts);
  float e = b + (npts > 0 ? (npts - 1) * delta : 0.0f);

  // ── Float header (indices 0-69) ──────────────────────────────────────────
  //   0 DELTA   1 DEPMIN  2 DEPMAX  5 B   6 E(SAC)  7 O
  //  56 DEPMEN 57 CMPAZ   58 CMPINC
  //  31 STLA   32 STLO  33 STEL  34 STDP  (left UNDEF – Cartesian coords)
  //  35 EVLA   36 EVLO  37 EVEL  38 EVDP  39 MAG   (left UNDEF)
  float fhdr[70];
  std::fill_n(fhdr, 70, UNDEF_F);
  fhdr[0] = delta; // DELTA [REQUIRED]
  fhdr[1] = depmin;
  fhdr[2] = depmax;
  fhdr[5] = b;       // B     [REQUIRED]
  fhdr[6] = e;       // E
  fhdr[7] = UNDEF_F; // O – event origin at reference time (Obspy leaves UNDEF)
  fhdr[56] = depmen;
  fhdr[57] = orientation.cmpaz;
  fhdr[58] = orientation.cmpinc;

  // ── Integer header (indices 0-39) ────────────────────────────────────────
  //   0-5  NZYEAR/JDAY/HOUR/MIN/SEC/MSEC   6 NVHDR   9 NPTS
  //  15 IFTYPE   16 IDEP   17 IZTYPE
  //  35 LEVEN    36 LPSPOL 37 LOVROK 38 LCALDA
  int32_t ihdr[40];
  std::fill_n(ihdr, 40, UNDEF_I);
  ihdr[0] = 1970;  // NZYEAR  (Unix epoch; no real event time in synthetics)
  ihdr[1] = 1;     // NZJDAY
  ihdr[2] = 0;     // NZHOUR
  ihdr[3] = 0;     // NZMIN
  ihdr[4] = 0;     // NZSEC
  ihdr[5] = 0;     // NZMSEC
  ihdr[6] = 6;     // NVHDR  (current SAC header version)
  ihdr[9] = npts;  // NPTS   [REQUIRED]
  ihdr[15] = 1;    // IFTYPE = ITIME (time-series)
  ihdr[16] = idep; // IDEP
  ihdr[17] = 11;   // IZTYPE = IORIGINT (origin time as reference)
  ihdr[35] = 1;    // LEVEN  (evenly-spaced) [REQUIRED]
  ihdr[36] = 0;    // LPSPOL (Obspy sets to 0)
  ihdr[37] = 1;    // LOVROK (overwrite OK)
  ihdr[38] = 1;    // LCALDA (auto-compute DIST/AZ/BAZ/GCARC)

  // ── Text header (192 bytes) ──────────────────────────────────────────────
  // Byte layout (each slot is 8 chars; KEVNM spans two consecutive slots):
  //   0  KSTNM(8)   8  KEVNM(16)  24  KHOLE(8)   32 KO   40 KA
  //  48  KT0-KT9(80)  128 KF  136 KUSER0(8)  144 KUSER1(8)  152 KUSER2(8)
  // 160  KCMPNM(8)  168 KNETWK(8)  176 KDATRD(8)  184 KINST(8)
  char thdr[192];
  for (int i = 0; i < 192; i += 8)
    std::memcpy(thdr + i, "-12345  ", 8); // SAC undefined-string sentinel

  fill_text_field(thdr + 0, station_name, 8);   // KSTNM
  fill_text_field(thdr + 8, "SPECFEMPP", 16);   // KEVNM
  fill_text_field(thdr + 24, location_code, 8); // KHOLE
  fill_text_field(thdr + 136, "SY", 8);         // KUSER0 (IRIS synthetic ID)
  fill_text_field(thdr + 144, "SPECFEMPP", 8);  // KUSER1 (code name)
  fill_text_field(thdr + 160, channel_code, 8); // KCMPNM
  fill_text_field(thdr + 168, network_name, 8); // KNETWK

  std::ofstream ofs(filepath, std::ios::binary);
  if (!ofs)
    throw std::runtime_error("Could not open SAC output file: " + filepath);

  ofs.write(reinterpret_cast<const char *>(fhdr), sizeof(fhdr));
  ofs.write(reinterpret_cast<const char *>(ihdr), sizeof(ihdr));
  ofs.write(thdr, 192);
  ofs.write(reinterpret_cast<const char *>(samples.data()),
            static_cast<std::streamsize>(npts * sizeof(float)));
}

// ---------------------------------------------------------------------------
/**
 * @brief SAC binary specialization of SeismogramFormatWriter.
 *
 * Writes one SAC binary (NVHDR=6) file per component for every
 * station / seismogram-type pair.  Output filenames are derived from the
 * SPECFEM++ channel-generator names (e.g. SY.STA01.S3.BXZ.semd) with the
 * time-series extension replaced by ".sac".
 *
 * SAC header fields populated:
 *   DELTA, B, O, NPTS, NVHDR, IFTYPE, IDEP, IZTYPE,
 *   LEVEN, LPSPOL, LOVROK, LCALDA, CMPAZ, CMPINC,
 *   KSTNM, KNETWK, KHOLE, KCMPNM, KEVNM, KUSER0, KUSER1.
 *
 * Geographic coordinate fields (STLA, STLO, EVLA, EVLO, …) are left at the
 * SAC undefined sentinel (−12345) because the data model uses Cartesian
 * coordinates rather than geographic ones.
 */
// ---------------------------------------------------------------------------
template <>
struct SeismogramFormatWriter<specfem::enums::seismogram_format::sac> {
  template <typename Receivers>
  static void write(Receivers &receivers, ChannelGenerator &gen,
                    const std::string &output_folder,
                    const std::optional<bool> write_from_main = std::nullopt) {
    const bool from_main = write_from_main.value_or(false);

    // SEED location code encodes simulation dimensionality.
    constexpr std::string_view location_code =
        (Receivers::dimension_tag == specfem::element::dimension_tag::dim2)
            ? "S2"
            : "S3";

    for (auto station_info : receivers.stations()) {
#ifdef SPECFEM_ENABLE_MPI
      if (station_info.islice != specfem::MPI::get_rank() &&
          !(from_main && specfem::MPI::main_proc())) {
        continue; // Skip stations not assigned to this rank
      }
#endif
      for (auto seismogram_type : station_info.get_seismogram_types()) {

        const auto base_filenames =
            gen.get_station_filenames<Receivers::dimension_tag>(
                station_info, seismogram_type);

        // SAC IDEP: 6=displacement, 7=velocity, 8=acceleration, −12345=other
        const int32_t idep = get_idep_for_wavefield(seismogram_type);

        // The channel generator is now dimension-aware and returns the same
        // number of filenames as value.size() for elastic types (2 for dim2,
        // 3 for dim3) and 1 for pressure.
        int ncomp = -1;
        float b = UNDEF_F;
        // Pre-seed delta from the configured timestep so that single-sample
        // seismograms still get a valid DELTA header value.
        float delta = static_cast<float>(gen.get_timestep());
        std::vector<std::vector<float> > samples;

        for (auto [time, value] : receivers.get_seismogram(
                 station_info.station_name, station_info.network_name,
                 seismogram_type)) {
          const float t = static_cast<float>(time);

          if (ncomp < 0) {
            // First sample: fix component count and begin time.
            ncomp = static_cast<int>(base_filenames.size());
            samples.resize(ncomp);
            b = t;
          } else if (samples[0].size() == 1) {
            // Second sample: derive the effective sample interval from data.
            delta = t - b;
          }

          for (int i = 0; i < ncomp; ++i)
            samples[i].push_back(static_cast<float>(value[i]));
        }

        if (ncomp <= 0 || samples.empty() || samples[0].empty())
          continue;

        // ── Write one SAC file per component ────────────────────────────────
        for (int icomp = 0; icomp < ncomp; ++icomp) {
          // Replace the last extension in the base filename with ".sac".
          const auto &base = base_filenames[icomp];
          const auto dot = base.rfind('.');
          const std::string stem =
              (dot == std::string::npos) ? base : base.substr(0, dot);
          const std::string sac_path = output_folder + "/" + stem + ".sac";

          // Extract channel code and orientation directly from the filename
          // to avoid hard-coded letter indices (correct for both 2-D and 3-D).
          const std::string channel_code =
              extract_channel_code_from_filename(base);
          const Orientation orientation = orientation_for_channel(channel_code);

          write_sac_binary(sac_path, station_info.station_name,
                           station_info.network_name, location_code,
                           channel_code, delta, b, orientation, idep,
                           samples[icomp]);
        }
      }
    }
  }
};

} // namespace impl
} // namespace io
} // namespace specfem
