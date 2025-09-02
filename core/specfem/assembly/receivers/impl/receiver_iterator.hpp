#pragma once

#include "enumerations/interface.hpp"
#include <vector>

namespace specfem::assembly::receivers_impl {

// Simple seismogram type iterator - just wraps a vector of seismogram types
class SeismogramTypeIterator {
public:
  using iterator = std::vector<specfem::wavefield::type>::const_iterator;

  SeismogramTypeIterator() = default;

  explicit SeismogramTypeIterator(
      const std::vector<specfem::wavefield::type> &types)
      : seismogram_types_(types) {}

  iterator begin() const { return seismogram_types_.begin(); }
  iterator end() const { return seismogram_types_.end(); }

  size_t size() const { return seismogram_types_.size(); }

private:
  std::vector<specfem::wavefield::type> seismogram_types_;
};

// StationInfo that contains station data and can provide seismogram types
struct StationInfo {
  std::string network_name;
  std::string station_name;

  StationInfo(std::string network, std::string station,
              const std::vector<specfem::wavefield::type> &types)
      : network_name(std::move(network)), station_name(std::move(station)),
        seismo_types_(types) {}

  // Method to get seismogram types associated with this station
  SeismogramTypeIterator get_seismogram_types() const {
    return SeismogramTypeIterator(seismo_types_);
  }

private:
  std::vector<specfem::wavefield::type> seismo_types_;
};

// Station iterator that outputs StationInfo objects
class StationIterator {
private:
  class Iterator {
  public:
    Iterator(const StationIterator *container, size_t index)
        : container_(container), index_(index) {}

    StationInfo operator*() const {
      return StationInfo(container_->network_names_[index_],
                         container_->station_names_[index_],
                         container_->seismogram_types_);
    }

    Iterator &operator++() {
      ++index_;
      return *this;
    }

    bool operator==(const Iterator &other) const {
      return index_ == other.index_;
    }

    bool operator!=(const Iterator &other) const {
      return index_ != other.index_;
    }

  private:
    const StationIterator *container_;
    size_t index_;
  };

public:
  StationIterator() = default;

  StationIterator(size_t nreceivers,
                  const std::vector<specfem::wavefield::type> &seismo_types)
      : seismogram_types_(seismo_types) {
    station_names_.reserve(nreceivers);
    network_names_.reserve(nreceivers);
  }

  Iterator begin() const { return Iterator(this, 0); }
  Iterator end() const { return Iterator(this, station_names_.size()); }

  size_t size() const { return station_names_.size(); }

protected:
  std::vector<std::string> station_names_;
  std::vector<std::string> network_names_;
  std::vector<specfem::wavefield::type> seismogram_types_;
};

// Primary template declaration for SeismogramIterator
template <specfem::dimension::type DimensionTag> class SeismogramIterator;

} // namespace specfem::assembly::receivers_impl

// Include dimension-specific seismogram iterator implementations
#include "../dim2/impl/seismogram_iterator.hpp"
#include "../dim3/impl/seismogram_iterator.hpp"
