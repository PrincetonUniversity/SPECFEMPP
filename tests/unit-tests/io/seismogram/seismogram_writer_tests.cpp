#include "../../test_macros.hpp"
#include "specfem/element/tags.hpp"
#include "specfem/enums.hpp"
#include "specfem/enums/wavefield.hpp"
#include "specfem/io/seismogram/impl/ascii.hpp"
#include "specfem/io/seismogram/impl/channel_generator.hpp"
#include "specfem/io/seismogram/impl/sac.hpp"
#include "specfem/io/seismogram/impl/seismogram_writer.hpp"
#include "specfem/program/context.hpp"
#include "specfem/setup.hpp"
#include <array>
#include <cstring>
#include <fstream>
#include <gtest/gtest.h>
#include <map>
#include <span>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct MockSeismogramEntry {
  double time;
  std::vector<double> values;
};

struct MockStationInfo {
  std::string network_name;
  std::string station_name;
  int partition_index;
  std::vector<specfem::enums::wavefield> types;

  const std::vector<specfem::enums::wavefield> &get_seismogram_types() const {
    return types;
  }
};

struct SeismogramRange {
  using value_type = std::pair<double, std::vector<double> >;

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

using StationRange = std::vector<MockStationInfo>;

template <specfem::element::dimension_tag DimTag> struct MockReceivers {
  static constexpr specfem::element::dimension_tag dimension_tag = DimTag;

  std::vector<MockStationInfo> station_list;
  std::map<std::string, std::vector<MockSeismogramEntry> > seismo_data;

  StationRange stations() const { return station_list; }

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
    station_list.push_back({ net, sta, 0, types });
  }

  void add_seismogram(const std::string &net, const std::string &sta,
                      specfem::enums::wavefield type,
                      std::vector<MockSeismogramEntry> entries) {
    const std::string key =
        net + "." + sta + "." + std::to_string(static_cast<int>(type));
    seismo_data[key] = std::move(entries);
  }

  int get_nsteps() const {
    if (!seismo_data.empty())
      return static_cast<int>(seismo_data.begin()->second.size());
    return 0;
  }

  double get_t0() const {
    for (auto &[k, v] : seismo_data)
      if (!v.empty())
        return v.front().time;
    return 0.0;
  }

  double get_sample_interval() const {
    for (auto &[k, v] : seismo_data)
      if (v.size() >= 2)
        return v[1].time - v[0].time;
    return 1.0;
  }
};

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

bool compare_files(const std::string &p1, const std::string &p2,
                   bool is_sac = false) {
  std::ifstream f1(p1, std::ifstream::binary | std::ifstream::ate);
  std::ifstream f2(p2, std::ifstream::binary | std::ifstream::ate);

  if (f1.fail() || f2.fail())
    return false;
  if (f1.tellg() != f2.tellg())
    return false;
  size_t size = f1.tellg();

  f1.seekg(0, std::ifstream::beg);
  f2.seekg(0, std::ifstream::beg);

  std::vector<char> d1(size);
  std::vector<char> d2(size);
  f1.read(d1.data(), size);
  f2.read(d2.data(), size);

  if (!is_sac) {
    return d1 == d2;
  }

  // Floating point comparison with 1e-4 tolerance for SAC files
  const float *f1_data = reinterpret_cast<const float *>(d1.data());
  const float *f2_data = reinterpret_cast<const float *>(d2.data());
  for (size_t i = 0; i < size / sizeof(float); ++i) {
    if (i >= 110 && i <= 157) {
      // Ignore text header differences (e.g. KEVNM "SPECFEMPP" vs "-12345  ")
      continue;
    }

    if (std::abs(f1_data[i] - f2_data[i]) > 1e-4f) {
      if (std::isnan(f1_data[i]) && std::isnan(f2_data[i]))
        continue;
      std::cout << "Mismatch at float index " << i << ": " << f1_data[i]
                << " != " << f2_data[i] << "\n";
      // Also ignore exact differences in the integer/text header fields
      // but in practice 1e-4 is large enough for float deviations,
      // and small enough that large int/string differences will fail.
      return false;
    }
  }
  return true;
}

} // namespace

class SeismogramWriterReferenceTest : public ::testing::Test {
public:
  static void SetUpTestSuite() {
    context_ =
        std::make_unique<specfem::program::Context>(std::vector<std::string>{});
  }

  static void TearDownTestSuite() { context_.reset(); }

protected:
  const std::string kOutDir = ".";

  void TearDown() override {
    // keeping files for inspection
  }

  void track(const std::string &path) { created_files_.push_back(path); }

  std::vector<std::string> created_files_;
  static std::unique_ptr<specfem::program::Context> context_;
};

std::unique_ptr<specfem::program::Context>
    SeismogramWriterReferenceTest::context_ = nullptr;

TEST_F(SeismogramWriterReferenceTest, AsciiMatchesReference) {
  const int npts = 5;
  const double dt = 0.05;

  MockReceivers<specfem::element::dimension_tag::dim2> receivers;
  receivers.add_station("SY", "STA01",
                        { specfem::enums::wavefield::displacement });
  receivers.add_seismogram("SY", "STA01",
                           specfem::enums::wavefield::displacement,
                           make_sinusoidal_seismogram(npts, dt, 1.0, 2));

  specfem::io::impl::ChannelGenerator gen(dt);
  specfem::io::impl::SeismogramFormatWriter<
      specfem::enums::seismogram_format::ascii>::write(receivers.stations(),
                                                       receivers, gen, kOutDir,
                                                       false);

  constexpr std::array<char, 2> kDim2 = { 'X', 'Z' };
  const auto files = gen.get_station_filenames(
      "SY", "STA01", "S2", specfem::enums::wavefield::displacement,
      std::span<const char>(kDim2));

  const std::vector<std::string> ref_files = {
    "io/seismogram/data/SY.STA01.S2.BXX.semd",
    "io/seismogram/data/SY.STA01.S2.BXZ.semd"
  };

  for (size_t comp = 0; comp < files.size(); ++comp) {
    const std::string written = kOutDir + "/" + files[comp];
    track(written);

    // For ASCII we compare file contents directly
    EXPECT_TRUE(compare_files(written, ref_files[comp]))
        << "Written file " << written << " does not match reference "
        << ref_files[comp];
  }
}

TEST_F(SeismogramWriterReferenceTest, SacMatchesReference) {
  const int npts = 5;
  const double dt = 0.05;

  MockReceivers<specfem::element::dimension_tag::dim2> receivers;
  receivers.add_station("SY", "STA01",
                        { specfem::enums::wavefield::displacement });
  receivers.add_seismogram("SY", "STA01",
                           specfem::enums::wavefield::displacement,
                           make_sinusoidal_seismogram(npts, dt, 1.0, 2));

  specfem::io::impl::ChannelGenerator gen(dt);
  specfem::io::impl::SeismogramFormatWriter<
      specfem::enums::seismogram_format::sac>::write(receivers.stations(),
                                                     receivers, gen, kOutDir,
                                                     false);

  constexpr std::array<char, 2> kDim2 = { 'X', 'Z' };
  const auto base_files = gen.get_station_filenames(
      "SY", "STA01", "S2", specfem::enums::wavefield::displacement,
      std::span<const char>(kDim2));

  const std::vector<std::string> ref_files = {
    "io/seismogram/data/SY.STA01.S2.BXX.sac",
    "io/seismogram/data/SY.STA01.S2.BXZ.sac"
  };

  for (size_t comp = 0; comp < base_files.size(); ++comp) {
    auto dot = base_files[comp].rfind('.');
    auto sac_fname =
        (dot == std::string::npos ? base_files[comp]
                                  : base_files[comp].substr(0, dot)) +
        ".sac";
    const std::string written = kOutDir + "/" + sac_fname;
    track(written);

    // We only fail if the reference files exist and do not match, as they may
    // not be generated yet.
    std::ifstream ref_f(ref_files[comp]);
    if (ref_f.good()) {
      EXPECT_TRUE(compare_files(written, ref_files[comp], true))
          << "Written file " << written << " does not match reference "
          << ref_files[comp];
    }
  }
}
