#ifndef _ENUMERATIONS_DIMENSION_HPP_
#define _ENUMERATIONS_DIMENSION_HPP_

#include "specfem_enums.hpp"

namespace specfem {
namespace dimension {

enum class type { dim2, dim3 };

template <specfem::dimension::type dimension> class dimension {
public:
  static constexpr specfem::dimension::type value = dimension;
  static constexpr int dim;
  static std::string to_string(){};
};

template <>
static constexpr int dimension<specfem::dimension::type::dim2>::dim = 2;

template <>
static constexpr int dimension<specfem::dimension::type::dim3>::dim = 3;

template <> std::string dimension<specfem::dimension::type::dim2>::to_string() {
  return "2D";
}

template <> std::string dimension<specfem::dimension::type::dim3>::to_string() {
  return "3D";
}

} // namespace dimension
} // namespace specfem

#endif /* _ENUMERATIONS_DIMENSION_HPP_ */
