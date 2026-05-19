#pragma once

#include "specfem/enums.hpp"
#include <algorithm>
#include <vector>

namespace specfem::assembly::receivers_impl {

// Simple seismogram type iterator - just wraps a vector of seismogram types
class SeismogramTypeIterator {
public:
  using iterator = std::vector<specfem::enums::wavefield>::const_iterator;

  SeismogramTypeIterator() = default;

  explicit SeismogramTypeIterator(
      const std::vector<specfem::enums::wavefield> &types)
      : seismogram_types_(types) {}

  iterator begin() const { return seismogram_types_.begin(); }
  iterator end() const { return seismogram_types_.end(); }

  size_t size() const { return seismogram_types_.size(); }

private:
  std::vector<specfem::enums::wavefield> seismogram_types_;
};

// StationInfo that contains station data and can provide seismogram types
struct StationInfo {
  std::string network_name;
  std::string station_name;
  int partition_index;

  StationInfo(std::string network, std::string station, int partition_index,
              const std::vector<specfem::enums::wavefield> &types)
      : network_name(std::move(network)), station_name(std::move(station)),
        partition_index(partition_index), seismo_types_(types) {}

  // Method to get seismogram types associated with this station
  SeismogramTypeIterator get_seismogram_types() const {
    return SeismogramTypeIterator(seismo_types_);
  }

private:
  std::vector<specfem::enums::wavefield> seismo_types_;
};

// Station iterator that outputs StationInfo objects
class StationIterator {
public:
  StationIterator() = default;

  StationIterator(size_t nreceivers,
                  const std::vector<specfem::enums::wavefield> &seismo_types)
      : seismogram_types_(seismo_types) {
    stations_.reserve(nreceivers);
  }

  auto begin() const { return stations_.begin(); }
  auto end() const { return stations_.end(); }

  size_t size() const { return stations_.size(); }

  /// Return a const ref to all stations.
  const std::vector<StationInfo> &stations() const { return stations_; }

  /// Return a vector of stations matching the predicate.
  template <typename Filter>
  std::vector<StationInfo> stations(Filter filter) const {
    std::vector<StationInfo> result;
    std::copy_if(stations_.begin(), stations_.end(), std::back_inserter(result),
                 filter);
    return result;
  }

protected:
  std::vector<StationInfo> stations_;
  std::vector<specfem::enums::wavefield> seismogram_types_;
};

} // namespace specfem::assembly::receivers_impl

// Unified seismogram iterator for all dimensions
#include "seismogram_iterator.hpp"
