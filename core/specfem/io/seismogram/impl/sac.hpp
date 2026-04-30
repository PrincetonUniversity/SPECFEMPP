#pragma once

#include "seismogram_writer.hpp"
#include "specfem/element/tags.hpp"
#include "specfem/enums.hpp"
#include "specfem/setup.hpp"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace specfem {
namespace io {
namespace impl {

// ---------------------------------------------------------------------------
// SAC binary format helpers
// ---------------------------------------------------------------------------
namespace sac_detail {

/**
 * @brief Write a single-component SAC binary (NVHDR=6) file.
 *
 * SAC binary layout (632-byte header + NPTS × float32 data):
 *   - 70 float32 fields  (280 bytes)
 *   - 40 int32   fields  (160 bytes)
 *   - 192 bytes  text fields
 *
 * Field order and indices follow the SAC file format specification:
 * https://ds.iris.edu/files/sac-manual/manual/file_format.html
 *
 * @param filepath        Absolute path of the output file.
 * @param station_name    SEED station code (≤8 chars, stored in KSTNM).
 * @param network_name    SEED network code (≤8 chars, stored in KNETWK).
 * @param location_code   SEED location ID (≤8 chars, stored in KHOLE).
 * @param channel_code    SEED channel code, e.g. "BXZ" (stored in KCMPNM).
 * @param delta           Seismogram sample interval [s] (DELTA).
 * @param b               Begin time of the trace relative to reference [s] (B).
 * @param cmpaz           Component azimuth   [deg, 0=N, 90=E] (CMPAZ).
 * @param cmpinc          Component inclination [deg, 0=up, 90=horiz] (CMPINC).
 * @param idep            SAC IDEP code (6=displ/nm, 7=vel/nm/s, 8=acc/nm/s²).
 * @param samples         Seismogram samples (values in SI units; SCALE=1).
 */
inline void write_sac_binary(const std::string &filepath,
                             const std::string &station_name,
                             const std::string &network_name,
                             const std::string &location_code,
                             const std::string &channel_code, const float delta,
                             const float b, const float cmpaz,
                             const float cmpinc, const int32_t idep,
                             const std::vector<float> &samples) {
  static constexpr float UNDEF_F = -12345.0f;
  static constexpr int32_t UNDEF_I = -12345;
  const int32_t npts = static_cast<int32_t>(samples.size());

  // ---- float header: 70 fields (indices 0-69) ----------------------------
  // Index mapping (0-based):
  //  0  DELTA      4  ODELTA     5  B      6  E (BYSAC)  7  O
  //  8  A          31 STLA      32 STLO   33 STEL        34 STDP
  //  35 EVLA       36 EVLO      37 EVEL   38 EVDP        39 MAG
  //  40-49 USER0-9  50 DIST      51 AZ    52 BAZ         53 GCARC
  //  56 DEPMEN     57 CMPAZ     58 CMPINC
  float fhdr[70];
  std::fill(fhdr, fhdr + 70, UNDEF_F);
  fhdr[0] = delta; // DELTA [REQUIRED]
  fhdr[5] = b;     // B – begin time relative to reference [REQUIRED]
  // fhdr[6] E is computed by SAC from B, DELTA, NPTS
  fhdr[7] = 0.0f; // O – event origin at reference time

  // Station coordinates: Cartesian → lat/lon unknown; leave as UNDEF.
  // (fhdr[31..34] STLA/STLO/STEL/STDP stay UNDEF_F)

  // Event coordinates: not available in current data model; leave as UNDEF.
  // (fhdr[35..39] EVLA/EVLO/EVEL/EVDP/MAG stay UNDEF_F)

  // DIST, AZ, BAZ, GCARC are computed by SAC when LCALDA=1 and STLA/EVLA
  // are defined; with undefined coordinates they remain UNDEF_F.

  fhdr[57] = cmpaz;  // CMPAZ
  fhdr[58] = cmpinc; // CMPINC

  // ---- integer header: 40 fields (indices 0-39) --------------------------
  // Index mapping (0-based):
  //  0 NZYEAR  1 NZJDAY  2 NZHOUR  3 NZMIN  4 NZSEC  5 NZMSEC
  //  6 NVHDR   7 NORID   8 NEVID   9 NPTS   10 INTERNAL
  //  11 NWFID  12 NXSIZE 13 NYSIZE 14 INTERNAL
  //  15 IFTYPE 16 IDEP   17 IZTYPE 18 INTERNAL 19 IINST
  //  20 ISTREG 21 IEVREG 22 IEVTYP 23 IQUAL  24 ISYNTH
  //  25 IMAGTYP 26 IMAGSRC 27-34 INTERNAL
  //  35 LEVEN  36 LPSPOL 37 LOVROK 38 LCALDA 39 INTERNAL
  int32_t ihdr[40];
  std::fill(ihdr, ihdr + 40, UNDEF_I);
  // Reference time: Unix epoch (synthetic, no real event time available)
  ihdr[0] = 1970;  // NZYEAR
  ihdr[1] = 1;     // NZJDAY
  ihdr[2] = 0;     // NZHOUR
  ihdr[3] = 0;     // NZMIN
  ihdr[4] = 0;     // NZSEC
  ihdr[5] = 0;     // NZMSEC
  ihdr[6] = 6;     // NVHDR (current SAC header version)
  ihdr[9] = npts;  // NPTS [REQUIRED]
  ihdr[15] = 1;    // IFTYPE = ITIME (time-series file)
  ihdr[16] = idep; // IDEP
  ihdr[17] = 11;   // IZTYPE = IORIGINT (origin time as reference)
  ihdr[35] = 1;    // LEVEN (evenly-spaced data) [REQUIRED]
  ihdr[36] = 1;    // LPSPOL (positive polarity)
  ihdr[37] = 1;    // LOVROK (overwrite OK)
  ihdr[38] = 1;    // LCALDA (compute DIST/AZ/BAZ/GCARC if STLA/EVLA set)

  // ---- text header: 192 bytes --------------------------------------------
  // Byte offsets within text section:
  //   0:   KSTNM   (8)   station name
  //   8:   KEVNM  (16)   event name
  //   24:  KHOLE   (8)   location ID (SEED)
  //   32:  KO      (8)   event origin label
  //   40:  KA      (8)   first arrival label
  //   48:  KT0-KT9 (10×8 = 80)  user picks
  //  128:  KF      (8)   end-of-event label
  //  136:  KUSER0  (8)
  //  144:  KUSER1  (8)
  //  152:  KUSER2  (8)
  //  160:  KCMPNM  (8)   channel code
  //  168:  KNETWK  (8)   network code
  //  176:  KDATRD  (8)
  //  184:  KINST   (8)
  char thdr[192];
  // Initialise all 8-char slots to the SAC undefined-string sentinel.
  // KEVNM occupies two consecutive 8-char slots (bytes 8-23).
  for (int i = 0; i < 192; i += 8) {
    std::memcpy(thdr + i, "-12345  ", 8);
  }

  // Helper: write string s left-justified into a field of `len` bytes,
  // padding with spaces on the right.
  auto fill_str = [](char *dst, const std::string &s, int len) {
    std::fill(dst, dst + len, ' ');
    int n = std::min(static_cast<int>(s.size()), len);
    std::memcpy(dst, s.data(), n);
  };

  fill_str(thdr + 0, station_name, 8);   // KSTNM
  fill_str(thdr + 8, "SPECFEMPP", 16);   // KEVNM (synthetics tag)
  fill_str(thdr + 24, location_code, 8); // KHOLE
  fill_str(thdr + 136, "SY", 8);         // KUSER0 (IRIS synthetic network ID)
  fill_str(thdr + 144, "SPECFEMPP", 8);  // KUSER1 (code name)
  fill_str(thdr + 160, channel_code, 8); // KCMPNM
  fill_str(thdr + 168, network_name, 8); // KNETWK

  // ---- write file --------------------------------------------------------
  std::ofstream ofs(filepath, std::ios::binary);
  if (!ofs.is_open()) {
    throw std::runtime_error("Could not open SAC output file: " + filepath);
  }
  ofs.write(reinterpret_cast<const char *>(fhdr), sizeof(fhdr));
  ofs.write(reinterpret_cast<const char *>(ihdr), sizeof(ihdr));
  ofs.write(thdr, 192);
  ofs.write(reinterpret_cast<const char *>(samples.data()),
            static_cast<std::streamsize>(npts * sizeof(float)));
}

/// Return the SAC IDEP code for a given wavefield type.
inline int32_t idep_for_wavefield(specfem::enums::wavefield type) {
  switch (type) {
  case specfem::enums::wavefield::displacement:
    return 6; // displacement / nm
  case specfem::enums::wavefield::velocity:
    return 7; // velocity / nm s⁻¹
  case specfem::enums::wavefield::acceleration:
    return 8; // acceleration / nm s⁻²
  default:
    return -12345; // undefined
  }
}

/// Return (CMPAZ, CMPINC) for component index in the X/Y/Z ordering.
/// X → East (horizontal), Y → North (horizontal), Z → Up (vertical).
inline std::pair<float, float> component_orientation(int icomp) {
  switch (icomp) {
  case 0:
    return { 90.0f, 90.0f }; // X – East, horizontal
  case 1:
    return { 0.0f, 90.0f }; // Y – North, horizontal
  case 2:
    return { 0.0f, 0.0f }; // Z – vertical (up)
  default:
    return { -12345.0f, -12345.0f };
  }
}

/// Replace the last dot-extension in a SPECFEM++ seismogram filename with
/// ".sac".  E.g. "SY.STA01.S3.BXZ.semd" → "SY.STA01.S3.BXZ.sac".
inline std::string to_sac_filename(const std::string &fname) {
  auto pos = fname.rfind('.');
  return (pos == std::string::npos) ? fname + ".sac"
                                    : fname.substr(0, pos) + ".sac";
}

} // namespace sac_detail

// ---------------------------------------------------------------------------
// SeismogramFormatWriter specialisation for SAC binary output
// ---------------------------------------------------------------------------

/**
 * @brief SAC binary specialisation of SeismogramFormatWriter.
 *
 * Writes one SAC binary (NVHDR=6) file per component for every
 * station/seismogram-type pair.  Files are named by replacing the SPECFEM++
 * time-series extension (e.g. .semd, .semv) with ".sac".
 *
 * SAC header fields populated:
 *   DELTA, B, O, NPTS, NVHDR, IFTYPE, IDEP, IZTYPE, LEVEN, LPSPOL, LOVROK,
 *   LCALDA, CMPAZ, CMPINC, KSTNM, KNETWK, KHOLE, KCMPNM, KEVNM, KUSER0,
 *   KUSER1.
 *
 * Fields requiring station/event geographic coordinates (STLA, STLO, EVLA,
 * EVLO, …) are set to the SAC undefined sentinel (−12345) because the
 * current data model stores Cartesian rather than geographic coordinates.
 */
template <>
struct SeismogramFormatWriter<specfem::enums::seismogram_format::sac> {
  template <typename Receivers>
  static void write(Receivers &receivers, ChannelGenerator &gen,
                    const std::string &output_folder) {
    using namespace sac_detail;

    // Derive the SEED location code from the simulation dimension.
    std::string location_code;
    if constexpr (Receivers::dimension_tag ==
                  specfem::element::dimension_tag::dim2) {
      location_code = "S2";
    } else {
      location_code = "S3";
    }

    for (auto station_info : receivers.stations()) {
      for (auto seismogram_type : station_info.get_seismogram_types()) {

        // Re-use the channel-generator filenames as the base naming source.
        // The SAC filename is obtained by replacing the extension with ".sac".
        const std::vector<std::string> ascii_filenames =
            gen.get_station_filenames<Receivers::dimension_tag>(
                station_info, seismogram_type);

        const int32_t idep = idep_for_wavefield(seismogram_type);

        // --- Collect seismogram samples (all components, all time steps) ---
        // ncomp is the minimum of the filename count and the actual value
        // array size to stay within bounds for 2-D (2-component) receivers
        // when the channel generator produces 3 filenames.
        int ncomp = -1;
        float b = -12345.0f;
        float delta = -12345.0f;
        std::vector<std::vector<float> > samples;

        int step = 0;
        for (auto [time, value] : receivers.get_seismogram(
                 station_info.station_name, station_info.network_name,
                 seismogram_type)) {
          const float t = static_cast<float>(time);

          if (ncomp < 0) {
            // First sample: initialise component count and begin time.
            ncomp = std::min(static_cast<int>(ascii_filenames.size()),
                             static_cast<int>(value.size()));
            samples.resize(ncomp);
            b = t;
          } else if (step == 1) {
            // Second sample: compute seismogram sample interval.
            delta = t - b;
          }

          for (int icomp = 0; icomp < ncomp; ++icomp) {
            samples[icomp].push_back(static_cast<float>(value[icomp]));
          }
          ++step;
        }

        if (ncomp <= 0 || samples.empty() || samples[0].empty()) {
          continue; // nothing to write
        }

        // --- Write one SAC file per component --------------------------------
        for (int icomp = 0; icomp < ncomp; ++icomp) {
          const std::string sac_path =
              output_folder + "/" + to_sac_filename(ascii_filenames[icomp]);

          // Channel code: derive from component letter (X/Y/Z or P).
          std::string channel_code;
          if (seismogram_type == specfem::enums::wavefield::pressure) {
            channel_code = gen.get_channel_code('P');
          } else {
            constexpr char letters[] = { 'X', 'Y', 'Z' };
            channel_code = gen.get_channel_code(letters[icomp % 3]);
          }

          // Component orientation (CMPAZ, CMPINC).
          auto [cmpaz, cmpinc] =
              (seismogram_type == specfem::enums::wavefield::pressure)
                  ? std::make_pair(-12345.0f, -12345.0f)
                  : component_orientation(icomp);

          write_sac_binary(sac_path, station_info.station_name,
                           station_info.network_name, location_code,
                           channel_code, delta, b, cmpaz, cmpinc, idep,
                           samples[icomp]);
        }
      }
    }
  }
};

} // namespace impl
} // namespace io
} // namespace specfem
