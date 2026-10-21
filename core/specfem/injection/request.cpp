#include "specfem/injection/request.hpp"
#include <algorithm>
#include <stdexcept>
#include <string>

const std::vector<spfp_quantity> &
specfem::injection::Request::quantities(const spfp_medium medium) const {
  switch (medium) {
  case spfp_medium_elastic:
    return elastic;
  case spfp_medium_acoustic:
    return acoustic;
  default:
    throw std::runtime_error("Unknown injection medium " +
                             std::to_string(medium) + ".");
  }
}

int specfem::injection::Request::components(const spfp_medium medium) const {
  return specfem::injection::count_components(quantities(medium));
}

int specfem::injection::Request::slot(const spfp_medium medium,
                                      const std::size_t index) const {
  const auto &list = quantities(medium);
  if (index >= list.size())
    throw std::out_of_range("Quantity index " + std::to_string(index) +
                            " is out of range for a request of " +
                            std::to_string(list.size()) + " quantities.");
  return specfem::injection::count_components(std::span(list).first(index));
}

int specfem::injection::Request::max_components() const {
  return std::max(components(spfp_medium_elastic),
                  components(spfp_medium_acoustic));
}
