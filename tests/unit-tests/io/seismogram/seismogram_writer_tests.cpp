/**
 * @file seismogram_writer_tests.cpp
 * @brief Unit tests for seismogram writing in ASCII and SAC formats.
 *
 * Tests cover:
 *  - ChannelGenerator: band codes, channel codes, filename generation,
 *    file extensions.
 *  - ASCII format writer: file creation, line count, value round-trip,
 *    output matches pre-baked reference files in data/.
 *  - SAC binary writer: file size, header fields, sample data round-trip.
 */

#include "../../test_macros.hpp"
#include "specfem/element/tags.hpp"
#include "specfem/enums.hpp"
#include "specfem/enums/wavefield.hpp"
#include "specfem/io/seismogram/impl/ascii.hpp"
#include "specfem/io/seismogram/impl/channel_generator.hpp"
#include "specfem/io/seismogram/impl/sac.hpp"
#include "specfem/io/seismogram/impl/seismogram_writer.hpp"
#include "specfem/setup.hpp"
#include <array>
#include <cstring>
#include <fstream>
#include <gtest/gtest.h>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

/// Count newline-terminated lines in a text file.
int count_lines(const std::string &path) {
  std::ifstream f(path);
  int n = 0;
  std::string line;
  while (std::getline(f, line))
    ++n;
  return n;
}

/// Remove a file if it exists.
void remove_if_exists(const std::string &path) { std::remove(path.c_str()); }

// ---------------------------------------------------------------------------
// Minimal mock Receivers
//
// Satisfies the duck-typed interface consumed by SeismogramFormatWriter:
//   - static constexpr dimension_tag
//   - stations()           → iterable of StationInfo-like objects
//   - get_seismogram(...)  → iterable of {time, std::vector<double>}
// ---------------------------------------------------------------------------

struct MockSeismogramEntry {
  double time;
  std::vector<double> values; // one entry per component
};

struct MockStationInfo {
  std::string network_name;
  std::string station_name;
  std::vector<specfem::enums::wavefield> types;

  // SeismogramTypeIterator interface (range-for over wavefield values)
  const std::vector<specfem::enums::wavefield> &get_seismogram_types() const {
    return types;
  }
};

/// Thin proxy returned by get_seismogram() to support range-for.
struct SeismogramRange {
  using value_type =
      std::pair<double, std::vector<double> >; // {time, value[ncomp]}

  const std::vector<MockSeismogramEntry> &entries;

  struct Iter {
    const std::vector<MockSeismogramEntry> *ptr;
    size_t idx;
    value_type operator*() const {
      return { ptr->at(idx).time, ptr->at(idx).values };
    }
    Iter &operator++() {
      ++idx;
      return *this;
    }
    bool operator!=(const Iter &o) const { return idx != o.idx; }
  };

  Iter begin() const { return { &entries, 0 }; }
  Iter end() const { return { &entries, entries.size() }; }
};

/// Iterable over MockStationInfo with range-for support.
struct StationRange {
  std::vector<MockStationInfo> stations_vec;
  auto begin() const { return stations_vec.begin(); }
  auto end() const { return stations_vec.end(); }
};

/**
 * @brief Mock Receivers object for seismogram format writer tests.
 *
 * @tparam DimTag Simulation dimension (dim2 or dim3).
 */
template <specfem::element::dimension_tag DimTag> struct MockReceivers {
  static constexpr specfem::element::dimension_tag dimension_tag = DimTag;

  std::vector<MockStationInfo> station_list;
  // Map key: "network.station.wavefield_ordinal"
  std::map<std::string, std::vector<MockSeismogramEntry> > seismo_data;

  StationRange stations() const { return { station_list }; }

  SeismogramRange get_seismogram(const std::string &station_name,
                                 const std::string &network_name,
                                 specfem::enums::wavefield type) const {
    const std::string key = network_name + "." + station_name + "." +
                            std::to_string(static_cast<int>(type));
    auto it = seismo_data.find(key);
    if (it == seismo_data.end()) {
      static const std::vector<MockSeismogramEntry> empty;
      return { empty };
    }
    return { it->second };
  }

  void add_station(const std::string &net, const std::string &sta,
                   const std::vector<specfem::enums::wavefield> &types) {
    station_list.push_back({ net, sta, types });
  }

  void add_seismogram(const std::string &net, const std::string &sta,
                      specfem::enums::wavefield type,
                      std::vector<MockSeismogramEntry> entries) {
    const std::string key =
        net + "." + sta + "." + std::to_string(static_cast<int>(type));
    seismo_data[key] = std::move(entries);
  }
};

/// Build a simple synthetic seismogram: value[c][i] = sin((c+1)*omega*t_i)
std::vector<MockSeismogramEntry>
make_sinusoidal_seismogram(int npts, double dt, double omega, int ncomp) {
  std::vector<MockSeismogramEntry> entries;
  entries.reserve(npts);
  for (int i = 0; i < npts; ++i) {
    MockSeismogramEntry e;
    e.time = i * dt;
    e.values.resize(ncomp);
    for (int c = 0; c < ncomp; ++c)
      e.values[c] = std::sin((c + 1) * omega * e.time);
    entries.push_back(e);
  }
  return entries;
}

// SAC binary layout constants (matches sac.hpp)
constexpr int kSACFloatHeaderBytes = 70 * sizeof(float); // 280
constexpr int kSACIntHeaderBytes = 40 * sizeof(int32_t); // 160
constexpr int kSACTextHeaderBytes = 192;
constexpr int kSACHeaderBytes =
    kSACFloatHeaderBytes + kSACIntHeaderBytes + kSACTextHeaderBytes; // 632

/// Read a float32 SAC header word at word index idx (0-based).
float read_sac_float(const std::string &path, int idx) {
  std::ifstream f(path, std::ios::binary);
  float val;
  f.seekg(idx * sizeof(float));
  f.read(reinterpret_cast<char *>(&val), sizeof(float));
  return val;
}

/// Read an int32 SAC header word at integer-header word index idx (0-based).
int32_t read_sac_int(const std::string &path, int idx) {
  std::ifstream f(path, std::ios::binary);
  int32_t val;
  f.seekg(kSACFloatHeaderBytes + idx * sizeof(int32_t));
  f.read(reinterpret_cast<char *>(&val), sizeof(int32_t));
  return val;
}

/// Read an 8-char text field from the SAC text header at byte offset.
std::string read_sac_text(const std::string &path, int byte_offset) {
  std::ifstream f(path, std::ios::binary);
  char buf[9] = {};
  f.seekg(kSACFloatHeaderBytes + kSACIntHeaderBytes + byte_offset);
  f.read(buf, 8);
  // Trim trailing spaces
  std::string s(buf);
  while (!s.empty() && s.back() == ' ')
    s.pop_back();
  return s;
}

/// Count the number of float32 data samples stored after the 632-byte header.
int read_sac_npts_from_filesize(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  const auto total = static_cast<int>(f.tellg());
  return (total - kSACHeaderBytes) / static_cast<int>(sizeof(float));
}

} // anonymous namespace

// ===========================================================================
// 1. ChannelGenerator tests
// ===========================================================================

class ChannelGeneratorTest : public ::testing::Test {
protected:
  using Gen = specfem::io::impl::ChannelGenerator;
};

TEST_F(ChannelGeneratorTest, BandCodeLongPeriod) {
  // dt >= 1.0 s → "L"
  Gen gen(1.0);
  EXPECT_EQ(gen.get_band_code(), "L");

  Gen gen2(5.0);
  EXPECT_EQ(gen2.get_band_code(), "L");
}

TEST_F(ChannelGeneratorTest, BandCodeMidPeriod) {
  // 0.1 <= dt < 1.0 → "M"
  Gen gen(0.1);
  EXPECT_EQ(gen.get_band_code(), "M");

  Gen gen2(0.5);
  EXPECT_EQ(gen2.get_band_code(), "M");
}

TEST_F(ChannelGeneratorTest, BandCodeBroadBand) {
  // 0.0125 < dt <= 0.1 → "B"  (boundary: dt=0.1 → M, not B)
  Gen gen(0.05); // 20 Hz → B
  EXPECT_EQ(gen.get_band_code(), "B");

  // dt slightly above 0.0125 → still B
  Gen gen2(0.013);
  EXPECT_EQ(gen2.get_band_code(), "B");
}

TEST_F(ChannelGeneratorTest, BandCodeHighBroadBand) {
  // 0.004 < dt <= 0.0125 → "H"
  Gen gen(0.01); // 100 Hz
  EXPECT_EQ(gen.get_band_code(), "H");

  Gen gen2(0.005);
  EXPECT_EQ(gen2.get_band_code(), "H");
}

TEST_F(ChannelGeneratorTest, BandCodeC) {
  // 0.001 < dt <= 0.004 → "C"
  Gen gen(0.002);
  EXPECT_EQ(gen.get_band_code(), "C");
}

TEST_F(ChannelGeneratorTest, BandCodeF) {
  // dt strictly < 0.001 → "F"
  // (avoid the exact 0.001 boundary which is ambiguous under float rounding)
  Gen gen(0.0009);
  EXPECT_EQ(gen.get_band_code(), "F");

  Gen gen2(0.0005);
  EXPECT_EQ(gen2.get_band_code(), "F");
}

TEST_F(ChannelGeneratorTest, ChannelCodeFormat) {
  // Band = 'B' (dt=0.05), instrument always 'X', orientation from argument.
  Gen gen(0.05);
  EXPECT_EQ(gen.get_channel_code('Z'), "BXZ");
  EXPECT_EQ(gen.get_channel_code('X'), "BXX");
  EXPECT_EQ(gen.get_channel_code('Y'), "BXY");
  EXPECT_EQ(gen.get_channel_code('P'), "BXP");
}

TEST_F(ChannelGeneratorTest, ChannelCodeVariesWithBand) {
  Gen genH(0.01); // H band
  EXPECT_EQ(genH.get_channel_code('Z'), "HXZ");

  Gen genL(2.0); // L band
  EXPECT_EQ(genL.get_channel_code('Z'), "LXZ");
}

TEST_F(ChannelGeneratorTest, FilenamesDisplacement3Components) {
  // dt=0.05 → band B; displacement → "semd"; 3 components X Y Z
  Gen gen(0.05);
  const auto files = gen.get_station_filenames(
      "SY", "STA01", "S3", specfem::enums::wavefield::displacement);

  ASSERT_EQ(files.size(), 3u);
  EXPECT_EQ(files[0], "SY.STA01.S3.BXX.semd");
  EXPECT_EQ(files[1], "SY.STA01.S3.BXY.semd");
  EXPECT_EQ(files[2], "SY.STA01.S3.BXZ.semd");
}

TEST_F(ChannelGeneratorTest, FilenamesVelocity) {
  Gen gen(0.05);
  const auto files = gen.get_station_filenames(
      "SY", "STA01", "S3", specfem::enums::wavefield::velocity);

  ASSERT_EQ(files.size(), 3u);
  EXPECT_EQ(files[0], "SY.STA01.S3.BXX.semv");
  EXPECT_EQ(files[1], "SY.STA01.S3.BXY.semv");
  EXPECT_EQ(files[2], "SY.STA01.S3.BXZ.semv");
}

TEST_F(ChannelGeneratorTest, FilenamesAcceleration) {
  Gen gen(0.05);
  const auto files = gen.get_station_filenames(
      "SY", "STA01", "S3", specfem::enums::wavefield::acceleration);

  ASSERT_EQ(files.size(), 3u);
  EXPECT_EQ(files[0], "SY.STA01.S3.BXX.sema");
}

TEST_F(ChannelGeneratorTest, FilenamePressure1Component) {
  Gen gen(0.05);
  const auto files = gen.get_station_filenames(
      "AC", "HYD01", "S2", specfem::enums::wavefield::pressure);

  ASSERT_EQ(files.size(), 1u);
  EXPECT_EQ(files[0], "AC.HYD01.S2.BXP.semp");
}

TEST_F(ChannelGeneratorTest, FilenamesEmptyLocationCode) {
  // Empty location code → the dot separator should be omitted.
  Gen gen(0.05);
  const auto files = gen.get_station_filenames(
      "SY", "STA01", "", specfem::enums::wavefield::displacement);

  ASSERT_EQ(files.size(), 3u);
  EXPECT_EQ(files[0], "SY.STA01.BXX.semd");
}

TEST_F(ChannelGeneratorTest, FileExtensions) {
  Gen gen(0.05);
  using wf = specfem::enums::wavefield;
  EXPECT_EQ(gen.get_file_extension(wf::displacement), "semd");
  EXPECT_EQ(gen.get_file_extension(wf::velocity), "semv");
  EXPECT_EQ(gen.get_file_extension(wf::acceleration), "sema");
  EXPECT_EQ(gen.get_file_extension(wf::pressure), "semp");
}

// ===========================================================================
// 2. write_sac_binary tests
// ===========================================================================

class WriteSACBinaryTest : public ::testing::Test {
protected:
  const std::string kTestDir = ".";

  std::string tmp_path(const std::string &name) {
    return kTestDir + "/" + name + ".sac";
  }

  void TearDown() override {
    for (const auto &f : created_files_)
      remove_if_exists(f);
  }

  std::string make_path(const std::string &name) {
    std::string p = tmp_path(name);
    created_files_.push_back(p);
    return p;
  }

  std::vector<std::string> created_files_;
};

TEST_F(WriteSACBinaryTest, FileIsCreated) {
  const std::string path = make_path("sac_created");
  std::vector<float> samples = { 1.0f, 2.0f, 3.0f };

  specfem::io::impl::write_sac_binary(path, "STA01", "SY", "S3", "BXZ", 0.01f,
                                      0.0f, { 0.0f, 0.0f }, // orientation
                                      6, samples);

  std::ifstream f(path, std::ios::binary);
  EXPECT_TRUE(f.good()) << "SAC file was not created at " << path;
}

TEST_F(WriteSACBinaryTest, FileSizeMatchesHeaderPlusSamples) {
  const int npts = 50;
  std::vector<float> samples(npts, 1.0f);
  const std::string path = make_path("sac_size");

  specfem::io::impl::write_sac_binary(path, "STA", "NET", "S3", "BXX", 0.05f,
                                      0.0f, { 90.0f, 90.0f }, 7, samples);

  const int expected_bytes =
      kSACHeaderBytes + npts * static_cast<int>(sizeof(float));
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  EXPECT_EQ(static_cast<int>(f.tellg()), expected_bytes);
}

TEST_F(WriteSACBinaryTest, NPTSFromFileSizeMatchesInput) {
  const int npts = 123;
  std::vector<float> samples(npts, 0.0f);
  const std::string path = make_path("sac_npts_size");

  specfem::io::impl::write_sac_binary(path, "S", "N", "", "BXZ", 0.01f, 0.0f,
                                      { 0.0f, 0.0f }, 6, samples);

  EXPECT_EQ(read_sac_npts_from_filesize(path), npts);
}

TEST_F(WriteSACBinaryTest, DeltaHeaderField) {
  const float delta = 0.025f;
  std::vector<float> samples = { 0.0f };
  const std::string path = make_path("sac_delta");

  specfem::io::impl::write_sac_binary(path, "S", "N", "", "BXZ", delta, 0.0f,
                                      { 0.0f, 0.0f }, 6, samples);

  // Float header word 0 = DELTA
  EXPECT_FLOAT_EQ(read_sac_float(path, 0), delta);
}

TEST_F(WriteSACBinaryTest, BeginTimeHeaderField) {
  const float b = 1.5f;
  std::vector<float> samples = { 0.0f };
  const std::string path = make_path("sac_btime");

  specfem::io::impl::write_sac_binary(path, "S", "N", "", "BXZ", 0.01f, b,
                                      { 0.0f, 0.0f }, 6, samples);

  // Float header word 5 = B
  EXPECT_FLOAT_EQ(read_sac_float(path, 5), b);
}

TEST_F(WriteSACBinaryTest, NPTSIntegerHeaderField) {
  const int npts = 77;
  std::vector<float> samples(npts, 0.0f);
  const std::string path = make_path("sac_npts_hdr");

  specfem::io::impl::write_sac_binary(path, "S", "N", "", "BXZ", 0.01f, 0.0f,
                                      { 0.0f, 0.0f }, 6, samples);

  // Integer header word 9 = NPTS
  EXPECT_EQ(read_sac_int(path, 9), npts);
}

TEST_F(WriteSACBinaryTest, NVHDRIsAlways6) {
  std::vector<float> samples = { 0.0f };
  const std::string path = make_path("sac_nvhdr");

  specfem::io::impl::write_sac_binary(path, "S", "N", "", "BXZ", 0.01f, 0.0f,
                                      { 0.0f, 0.0f }, 6, samples);

  // Integer header word 6 = NVHDR, must be 6
  EXPECT_EQ(read_sac_int(path, 6), 6);
}

TEST_F(WriteSACBinaryTest, IFTYPEIsITIME) {
  std::vector<float> samples = { 0.0f };
  const std::string path = make_path("sac_iftype");

  specfem::io::impl::write_sac_binary(path, "S", "N", "", "BXZ", 0.01f, 0.0f,
                                      { 0.0f, 0.0f }, 6, samples);

  // Integer header word 15 = IFTYPE; 1 = ITIME
  EXPECT_EQ(read_sac_int(path, 15), 1);
}

TEST_F(WriteSACBinaryTest, IDEPHeaderField) {
  const int32_t idep = 7; // velocity
  std::vector<float> samples = { 0.0f };
  const std::string path = make_path("sac_idep");

  specfem::io::impl::write_sac_binary(path, "S", "N", "", "BXZ", 0.01f, 0.0f,
                                      { 0.0f, 0.0f }, idep, samples);

  // Integer header word 16 = IDEP
  EXPECT_EQ(read_sac_int(path, 16), idep);
}

TEST_F(WriteSACBinaryTest, LEVENIsSet) {
  std::vector<float> samples = { 0.0f };
  const std::string path = make_path("sac_leven");

  specfem::io::impl::write_sac_binary(path, "S", "N", "", "BXZ", 0.01f, 0.0f,
                                      { 0.0f, 0.0f }, 6, samples);

  // Integer header word 35 = LEVEN; 1 = evenly spaced
  EXPECT_EQ(read_sac_int(path, 35), 1);
}

TEST_F(WriteSACBinaryTest, ComponentOrientationCMPAZ) {
  const specfem::io::impl::Orientation orient = { 45.0f, 90.0f };
  std::vector<float> samples = { 0.0f };
  const std::string path = make_path("sac_cmpaz");

  specfem::io::impl::write_sac_binary(path, "S", "N", "", "BXX", 0.01f, 0.0f,
                                      orient, 6, samples);

  // Float header word 57 = CMPAZ
  EXPECT_FLOAT_EQ(read_sac_float(path, 57), orient.cmpaz);
  // Float header word 58 = CMPINC
  EXPECT_FLOAT_EQ(read_sac_float(path, 58), orient.cmpinc);
}

TEST_F(WriteSACBinaryTest, StationNameInTextHeader) {
  std::vector<float> samples = { 0.0f };
  const std::string path = make_path("sac_kstnm");

  specfem::io::impl::write_sac_binary(path, "MYSTA", "SY", "S3", "BXZ", 0.01f,
                                      0.0f, { 0.0f, 0.0f }, 6, samples);

  // Text header byte 0 = KSTNM (8 chars)
  EXPECT_EQ(read_sac_text(path, 0), "MYSTA");
}

TEST_F(WriteSACBinaryTest, NetworkNameInTextHeader) {
  std::vector<float> samples = { 0.0f };
  const std::string path = make_path("sac_knetwk");

  specfem::io::impl::write_sac_binary(path, "STA", "MYNET", "S3", "BXZ", 0.01f,
                                      0.0f, { 0.0f, 0.0f }, 6, samples);

  // Text header byte 168 = KNETWK (8 chars)
  EXPECT_EQ(read_sac_text(path, 168), "MYNET");
}

TEST_F(WriteSACBinaryTest, ChannelCodeInTextHeader) {
  std::vector<float> samples = { 0.0f };
  const std::string path = make_path("sac_kcmpnm");

  specfem::io::impl::write_sac_binary(path, "STA", "NET", "S3", "BXZ", 0.01f,
                                      0.0f, { 0.0f, 0.0f }, 6, samples);

  // Text header byte 160 = KCMPNM (8 chars)
  EXPECT_EQ(read_sac_text(path, 160), "BXZ");
}

TEST_F(WriteSACBinaryTest, SamplesRoundTrip) {
  const std::vector<float> original = { 1.0f, -2.5f, 3.14f, 0.0f, -99.9f };
  const std::string path = make_path("sac_samples");

  specfem::io::impl::write_sac_binary(path, "S", "N", "", "BXZ", 0.01f, 0.0f,
                                      { 0.0f, 0.0f }, 6, original);

  // Read raw samples back from after the 632-byte header
  std::ifstream f(path, std::ios::binary);
  f.seekg(kSACHeaderBytes);
  std::vector<float> read_back(original.size());
  f.read(reinterpret_cast<char *>(read_back.data()),
         static_cast<std::streamsize>(original.size() * sizeof(float)));

  ASSERT_EQ(read_back.size(), original.size());
  for (size_t i = 0; i < original.size(); ++i) {
    EXPECT_FLOAT_EQ(read_back[i], original[i]) << "Mismatch at sample " << i;
  }
}

TEST_F(WriteSACBinaryTest, EmptySamplesCreatesHeaderOnlyFile) {
  // An empty seismogram should still produce a valid 632-byte file.
  const std::string path = make_path("sac_empty");
  std::vector<float> empty;

  specfem::io::impl::write_sac_binary(path, "S", "N", "", "BXZ", 0.01f, 0.0f,
                                      { 0.0f, 0.0f }, 6, empty);

  std::ifstream f(path, std::ios::binary | std::ios::ate);
  EXPECT_EQ(static_cast<int>(f.tellg()), kSACHeaderBytes);
}

// ===========================================================================
// 3. ASCII format writer tests
// ===========================================================================

class ASCIISeismogramWriterTest : public ::testing::Test {
protected:
  const std::string kOutDir = ".";

  void TearDown() override {
    for (const auto &f : created_files_)
      remove_if_exists(f);
  }

  void track(const std::string &path) { created_files_.push_back(path); }

  std::vector<std::string> created_files_;
};

TEST_F(ASCIISeismogramWriterTest, FilesCreatedForElasticDisplacement) {
  // 2D receiver, displacement → 3 component files (X, Y, Z)
  MockReceivers<specfem::element::dimension_tag::dim2> receivers;
  receivers.add_station("SY", "STA01",
                        { specfem::enums::wavefield::displacement });

  const int npts = 10;
  receivers.add_seismogram("SY", "STA01",
                           specfem::enums::wavefield::displacement,
                           make_sinusoidal_seismogram(npts, 0.05, 1.0, 3));

  specfem::io::impl::ChannelGenerator gen(0.05); // Band B

  specfem::io::impl::SeismogramFormatWriter<
      specfem::enums::seismogram_format::ascii>::write(receivers, gen, kOutDir);

  const auto files = gen.get_station_filenames(
      "SY", "STA01", "S2", specfem::enums::wavefield::displacement);
  ASSERT_EQ(files.size(), 3u);

  for (const auto &fname : files) {
    const std::string path = kOutDir + "/" + fname;
    track(path);
    std::ifstream f(path);
    EXPECT_TRUE(f.good()) << "Missing file: " << path;
  }
}

TEST_F(ASCIISeismogramWriterTest, LineCountMatchesNPTS) {
  // Each simulation step produces one line per component.
  MockReceivers<specfem::element::dimension_tag::dim2> receivers;
  receivers.add_station("SY", "STA01", { specfem::enums::wavefield::velocity });

  const int npts = 25;
  receivers.add_seismogram("SY", "STA01", specfem::enums::wavefield::velocity,
                           make_sinusoidal_seismogram(npts, 0.01, 2.0, 3));

  specfem::io::impl::ChannelGenerator gen(0.01);
  specfem::io::impl::SeismogramFormatWriter<
      specfem::enums::seismogram_format::ascii>::write(receivers, gen, kOutDir);

  const auto files = gen.get_station_filenames(
      "SY", "STA01", "S2", specfem::enums::wavefield::velocity);
  for (const auto &fname : files) {
    const std::string path = kOutDir + "/" + fname;
    track(path);
    EXPECT_EQ(count_lines(path), npts) << "Wrong line count for " << fname;
  }
}

TEST_F(ASCIISeismogramWriterTest, ValuesRoundTripForAllComponents) {
  // Write known values, read them back and compare.
  const int npts = 5;
  const double dt = 0.1;

  MockReceivers<specfem::element::dimension_tag::dim3> receivers;
  receivers.add_station("II", "ANMO",
                        { specfem::enums::wavefield::displacement });

  auto entries =
      make_sinusoidal_seismogram(npts, dt, /*omega=*/1.0, /*ncomp=*/3);
  receivers.add_seismogram("II", "ANMO",
                           specfem::enums::wavefield::displacement, entries);

  specfem::io::impl::ChannelGenerator gen(dt);
  specfem::io::impl::SeismogramFormatWriter<
      specfem::enums::seismogram_format::ascii>::write(receivers, gen, kOutDir);

  const auto files = gen.get_station_filenames(
      "II", "ANMO", "S3", specfem::enums::wavefield::displacement);
  ASSERT_EQ(files.size(), 3u);

  for (int comp = 0; comp < 3; ++comp) {
    const std::string path = kOutDir + "/" + files[comp];
    track(path);

    std::ifstream f(path);
    ASSERT_TRUE(f.good()) << "Cannot open " << path;

    for (int i = 0; i < npts; ++i) {
      double t_read, v_read;
      f >> t_read >> v_read;
      EXPECT_NEAR(t_read, entries[i].time, 1e-12)
          << "time mismatch at step " << i << " comp " << comp;
      // ASCII scientific-notation output has ~8 significant digits; use a
      // tolerance that accommodates the formatting precision loss.
      EXPECT_NEAR(v_read, entries[i].values[comp], 1e-7)
          << "value mismatch at step " << i << " comp " << comp;
    }
  }
}

TEST_F(ASCIISeismogramWriterTest, PressureWritesSingleFile) {
  MockReceivers<specfem::element::dimension_tag::dim2> receivers;
  receivers.add_station("AC", "HYD01", { specfem::enums::wavefield::pressure });

  const int npts = 8;
  receivers.add_seismogram("AC", "HYD01", specfem::enums::wavefield::pressure,
                           make_sinusoidal_seismogram(npts, 0.05, 1.0, 1));

  specfem::io::impl::ChannelGenerator gen(0.05);
  specfem::io::impl::SeismogramFormatWriter<
      specfem::enums::seismogram_format::ascii>::write(receivers, gen, kOutDir);

  const auto files = gen.get_station_filenames(
      "AC", "HYD01", "S2", specfem::enums::wavefield::pressure);
  ASSERT_EQ(files.size(), 1u);

  const std::string path = kOutDir + "/" + files[0];
  track(path);
  std::ifstream f(path);
  EXPECT_TRUE(f.good());
  EXPECT_EQ(count_lines(path), npts);
}

TEST_F(ASCIISeismogramWriterTest, MultipleStationsWriteSeparateFiles) {
  MockReceivers<specfem::element::dimension_tag::dim2> receivers;
  receivers.add_station("SY", "STA01",
                        { specfem::enums::wavefield::displacement });
  receivers.add_station("SY", "STA02",
                        { specfem::enums::wavefield::displacement });

  const int npts = 5;
  for (const auto &sta : { std::string("STA01"), std::string("STA02") }) {
    receivers.add_seismogram("SY", sta, specfem::enums::wavefield::displacement,
                             make_sinusoidal_seismogram(npts, 0.05, 1.0, 3));
  }

  specfem::io::impl::ChannelGenerator gen(0.05);
  specfem::io::impl::SeismogramFormatWriter<
      specfem::enums::seismogram_format::ascii>::write(receivers, gen, kOutDir);

  for (const auto &sta : { std::string("STA01"), std::string("STA02") }) {
    const auto files = gen.get_station_filenames(
        "SY", sta, "S2", specfem::enums::wavefield::displacement);
    for (const auto &fname : files) {
      const std::string path = kOutDir + "/" + fname;
      track(path);
      EXPECT_TRUE(std::ifstream(path).good()) << "Missing file: " << path;
    }
  }
}

/**
 * @brief Verify the exact byte-for-byte ASCII output against pre-baked
 *        reference files stored in io/seismogram/data/.
 *
 * Reference files were generated with:
 *   network=SY, station=STA01, dt=0.05 (band B), omega=1.0, npts=5, dim2
 *   value[c][i] = sin((c+1) * 1.0 * i * 0.05)
 * and written with std::scientific (default precision = 6 decimal digits).
 */
TEST_F(ASCIISeismogramWriterTest, OutputMatchesReferenceFiles) {
  const int npts = 5;
  const double dt = 0.05;

  MockReceivers<specfem::element::dimension_tag::dim2> receivers;
  receivers.add_station("SY", "STA01",
                        { specfem::enums::wavefield::displacement });
  receivers.add_seismogram("SY", "STA01",
                           specfem::enums::wavefield::displacement,
                           make_sinusoidal_seismogram(npts, dt, 1.0, 3));

  specfem::io::impl::ChannelGenerator gen(dt); // dt=0.05 → band B

  specfem::io::impl::SeismogramFormatWriter<
      specfem::enums::seismogram_format::ascii>::write(receivers, gen, kOutDir);

  const auto files = gen.get_station_filenames(
      "SY", "STA01", "S2", specfem::enums::wavefield::displacement);
  ASSERT_EQ(files.size(), 3u);

  // Reference files live in io/seismogram/data/ relative to the working dir
  const std::vector<std::string> ref_files = {
    "io/seismogram/data/SY.STA01.S2.BXX.semd",
    "io/seismogram/data/SY.STA01.S2.BXY.semd",
    "io/seismogram/data/SY.STA01.S2.BXZ.semd"
  };

  for (int comp = 0; comp < 3; ++comp) {
    const std::string written = kOutDir + "/" + files[comp];
    track(written);

    std::ifstream written_f(written);
    std::ifstream ref_f(ref_files[comp]);
    ASSERT_TRUE(written_f.good()) << "Written file missing: " << written;
    ASSERT_TRUE(ref_f.good())
        << "Reference file missing: " << ref_files[comp]
        << "  (run tests with WORKING_DIRECTORY containing io/)";

    std::string written_line, ref_line;
    int n = 0;
    while (std::getline(written_f, written_line) &&
           std::getline(ref_f, ref_line)) {
      EXPECT_EQ(written_line, ref_line)
          << "Line " << n << " differs for component " << comp;
      ++n;
    }
    EXPECT_EQ(n, npts) << "Unexpected number of lines for component " << comp;
  }
}

// ===========================================================================
// 4. SAC format writer tests
// ===========================================================================

class SACSeismogramWriterTest : public ::testing::Test {
protected:
  const std::string kOutDir = ".";

  void TearDown() override {
    for (const auto &f : created_files_)
      remove_if_exists(f);
  }

  void track(const std::string &path) { created_files_.push_back(path); }

  // Derive the expected SAC filename from a base ASCII filename.
  static std::string ascii_to_sac(const std::string &base) {
    const auto dot = base.rfind('.');
    return (dot == std::string::npos ? base : base.substr(0, dot)) + ".sac";
  }

  std::vector<std::string> created_files_;
};

TEST_F(SACSeismogramWriterTest, FilesCreatedForElasticDisplacement) {
  MockReceivers<specfem::element::dimension_tag::dim3> receivers;
  receivers.add_station("SY", "STA01",
                        { specfem::enums::wavefield::displacement });

  const int npts = 10;
  receivers.add_seismogram("SY", "STA01",
                           specfem::enums::wavefield::displacement,
                           make_sinusoidal_seismogram(npts, 0.05, 1.0, 3));

  specfem::io::impl::ChannelGenerator gen(0.05);
  specfem::io::impl::SeismogramFormatWriter<
      specfem::enums::seismogram_format::sac>::write(receivers, gen, kOutDir);

  const auto base_files = gen.get_station_filenames(
      "SY", "STA01", "S3", specfem::enums::wavefield::displacement);
  ASSERT_EQ(base_files.size(), 3u);

  for (const auto &fname : base_files) {
    const std::string path = kOutDir + "/" + ascii_to_sac(fname);
    track(path);
    std::ifstream f(path, std::ios::binary);
    EXPECT_TRUE(f.good()) << "Missing SAC file: " << path;
  }
}

TEST_F(SACSeismogramWriterTest, FileSizesCorrect) {
  const int npts = 20;
  MockReceivers<specfem::element::dimension_tag::dim3> receivers;
  receivers.add_station("SY", "STA01", { specfem::enums::wavefield::velocity });
  receivers.add_seismogram("SY", "STA01", specfem::enums::wavefield::velocity,
                           make_sinusoidal_seismogram(npts, 0.01, 3.0, 3));

  specfem::io::impl::ChannelGenerator gen(0.01);
  specfem::io::impl::SeismogramFormatWriter<
      specfem::enums::seismogram_format::sac>::write(receivers, gen, kOutDir);

  const auto base_files = gen.get_station_filenames(
      "SY", "STA01", "S3", specfem::enums::wavefield::velocity);
  const int expected_size =
      kSACHeaderBytes + npts * static_cast<int>(sizeof(float));

  for (const auto &fname : base_files) {
    const std::string path = kOutDir + "/" + ascii_to_sac(fname);
    track(path);
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    EXPECT_EQ(static_cast<int>(f.tellg()), expected_size)
        << "Wrong file size for " << path;
  }
}

TEST_F(SACSeismogramWriterTest, DeltaHeaderCorrect) {
  const double dt = 0.05;
  const int npts = 5;
  MockReceivers<specfem::element::dimension_tag::dim3> receivers;
  receivers.add_station("SY", "STA01",
                        { specfem::enums::wavefield::displacement });
  receivers.add_seismogram("SY", "STA01",
                           specfem::enums::wavefield::displacement,
                           make_sinusoidal_seismogram(npts, dt, 1.0, 3));

  specfem::io::impl::ChannelGenerator gen(dt);
  specfem::io::impl::SeismogramFormatWriter<
      specfem::enums::seismogram_format::sac>::write(receivers, gen, kOutDir);

  const auto base_files = gen.get_station_filenames(
      "SY", "STA01", "S3", specfem::enums::wavefield::displacement);
  for (const auto &fname : base_files) {
    const std::string path = kOutDir + "/" + ascii_to_sac(fname);
    track(path);
    // DELTA = second sample time minus begin time = dt
    EXPECT_NEAR(read_sac_float(path, 0), static_cast<float>(dt), 1e-5f)
        << "DELTA mismatch for " << path;
  }
}

TEST_F(SACSeismogramWriterTest, NPTSHeaderCorrect) {
  const int npts = 33;
  MockReceivers<specfem::element::dimension_tag::dim3> receivers;
  receivers.add_station("SY", "STA01",
                        { specfem::enums::wavefield::acceleration });
  receivers.add_seismogram("SY", "STA01",
                           specfem::enums::wavefield::acceleration,
                           make_sinusoidal_seismogram(npts, 0.05, 1.0, 3));

  specfem::io::impl::ChannelGenerator gen(0.05);
  specfem::io::impl::SeismogramFormatWriter<
      specfem::enums::seismogram_format::sac>::write(receivers, gen, kOutDir);

  const auto base_files = gen.get_station_filenames(
      "SY", "STA01", "S3", specfem::enums::wavefield::acceleration);
  for (const auto &fname : base_files) {
    const std::string path = kOutDir + "/" + ascii_to_sac(fname);
    track(path);
    EXPECT_EQ(read_sac_int(path, 9), npts) << "NPTS mismatch for " << path;
  }
}

TEST_F(SACSeismogramWriterTest, IDEPIsDisplacementFor6) {
  // displacement → IDEP = 6
  const int npts = 5;
  MockReceivers<specfem::element::dimension_tag::dim3> receivers;
  receivers.add_station("SY", "STA01",
                        { specfem::enums::wavefield::displacement });
  receivers.add_seismogram("SY", "STA01",
                           specfem::enums::wavefield::displacement,
                           make_sinusoidal_seismogram(npts, 0.05, 1.0, 3));

  specfem::io::impl::ChannelGenerator gen(0.05);
  specfem::io::impl::SeismogramFormatWriter<
      specfem::enums::seismogram_format::sac>::write(receivers, gen, kOutDir);

  const auto base_files = gen.get_station_filenames(
      "SY", "STA01", "S3", specfem::enums::wavefield::displacement);
  for (const auto &fname : base_files) {
    const std::string path = kOutDir + "/" + ascii_to_sac(fname);
    track(path);
    EXPECT_EQ(read_sac_int(path, 16), 6) << "IDEP should be 6 for displacement";
  }
}

TEST_F(SACSeismogramWriterTest, IDEPIsVelocityFor7) {
  const int npts = 5;
  MockReceivers<specfem::element::dimension_tag::dim3> receivers;
  receivers.add_station("SY", "STA01", { specfem::enums::wavefield::velocity });
  receivers.add_seismogram("SY", "STA01", specfem::enums::wavefield::velocity,
                           make_sinusoidal_seismogram(npts, 0.05, 1.0, 3));

  specfem::io::impl::ChannelGenerator gen(0.05);
  specfem::io::impl::SeismogramFormatWriter<
      specfem::enums::seismogram_format::sac>::write(receivers, gen, kOutDir);

  const auto base_files = gen.get_station_filenames(
      "SY", "STA01", "S3", specfem::enums::wavefield::velocity);
  for (const auto &fname : base_files) {
    const std::string path = kOutDir + "/" + ascii_to_sac(fname);
    track(path);
    EXPECT_EQ(read_sac_int(path, 16), 7) << "IDEP should be 7 for velocity";
  }
}

TEST_F(SACSeismogramWriterTest, StationNameStoredInHeader) {
  const int npts = 5;
  MockReceivers<specfem::element::dimension_tag::dim3> receivers;
  receivers.add_station("SY", "MYSTA",
                        { specfem::enums::wavefield::displacement });
  receivers.add_seismogram("SY", "MYSTA",
                           specfem::enums::wavefield::displacement,
                           make_sinusoidal_seismogram(npts, 0.05, 1.0, 3));

  specfem::io::impl::ChannelGenerator gen(0.05);
  specfem::io::impl::SeismogramFormatWriter<
      specfem::enums::seismogram_format::sac>::write(receivers, gen, kOutDir);

  const auto base_files = gen.get_station_filenames(
      "SY", "MYSTA", "S3", specfem::enums::wavefield::displacement);
  const std::string path = kOutDir + "/" + ascii_to_sac(base_files[0]);
  track(path);

  EXPECT_EQ(read_sac_text(path, 0), "MYSTA"); // KSTNM
}

TEST_F(SACSeismogramWriterTest, NetworkNameStoredInHeader) {
  const int npts = 5;
  MockReceivers<specfem::element::dimension_tag::dim3> receivers;
  receivers.add_station("MYNET", "STA01",
                        { specfem::enums::wavefield::displacement });
  receivers.add_seismogram("MYNET", "STA01",
                           specfem::enums::wavefield::displacement,
                           make_sinusoidal_seismogram(npts, 0.05, 1.0, 3));

  specfem::io::impl::ChannelGenerator gen(0.05);
  specfem::io::impl::SeismogramFormatWriter<
      specfem::enums::seismogram_format::sac>::write(receivers, gen, kOutDir);

  const auto base_files = gen.get_station_filenames(
      "MYNET", "STA01", "S3", specfem::enums::wavefield::displacement);
  const std::string path = kOutDir + "/" + ascii_to_sac(base_files[0]);
  track(path);

  EXPECT_EQ(read_sac_text(path, 168), "MYNET"); // KNETWK
}

TEST_F(SACSeismogramWriterTest, SamplesDataRoundTrip) {
  // Write a 3-component displacement seismogram; verify float32 samples.
  const int npts = 6;
  const double dt = 0.05;
  MockReceivers<specfem::element::dimension_tag::dim3> receivers;
  receivers.add_station("SY", "STA01",
                        { specfem::enums::wavefield::displacement });

  auto entries =
      make_sinusoidal_seismogram(npts, dt, /*omega=*/2.0, /*ncomp=*/3);
  receivers.add_seismogram("SY", "STA01",
                           specfem::enums::wavefield::displacement, entries);

  specfem::io::impl::ChannelGenerator gen(dt);
  specfem::io::impl::SeismogramFormatWriter<
      specfem::enums::seismogram_format::sac>::write(receivers, gen, kOutDir);

  const auto base_files = gen.get_station_filenames(
      "SY", "STA01", "S3", specfem::enums::wavefield::displacement);

  for (int comp = 0; comp < 3; ++comp) {
    const std::string path = kOutDir + "/" + ascii_to_sac(base_files[comp]);
    track(path);

    std::ifstream f(path, std::ios::binary);
    ASSERT_TRUE(f.good());
    f.seekg(kSACHeaderBytes);
    std::vector<float> read_back(npts);
    f.read(reinterpret_cast<char *>(read_back.data()),
           static_cast<std::streamsize>(npts * sizeof(float)));

    for (int i = 0; i < npts; ++i) {
      EXPECT_NEAR(read_back[i], static_cast<float>(entries[i].values[comp]),
                  1e-5f)
          << "Sample mismatch comp=" << comp << " step=" << i;
    }
  }
}

TEST_F(SACSeismogramWriterTest, EmptySeismogramProducesNoFile) {
  // If get_seismogram() returns no samples, the SAC writer should skip the
  // station entirely rather than producing a zero-size or corrupt file.
  MockReceivers<specfem::element::dimension_tag::dim3> receivers;
  receivers.add_station("SY", "STA01",
                        { specfem::enums::wavefield::displacement });
  // Intentionally add NO seismogram data for STA01

  specfem::io::impl::ChannelGenerator gen(0.05);
  specfem::io::impl::SeismogramFormatWriter<
      specfem::enums::seismogram_format::sac>::write(receivers, gen, kOutDir);

  const auto base_files = gen.get_station_filenames(
      "SY", "STA01", "S3", specfem::enums::wavefield::displacement);
  for (const auto &fname : base_files) {
    const std::string path = kOutDir + "/" + ascii_to_sac(fname);
    // The file should NOT exist because ncomp <= 0 check in sac.hpp skips it.
    EXPECT_FALSE(std::ifstream(path, std::ios::binary).good())
        << "Should not have created file for empty seismogram: " << path;
  }
}
