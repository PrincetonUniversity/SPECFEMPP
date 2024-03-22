#ifndef _ENUMERATIONS_DIMENSION_HPP_
#define _ENUMERATIONS_DIMENSION_HPP_

#include "specfem_enums.hpp"

namespace specfem {
namespace dimension {

enum class type { dim2, dim3 };

template <specfem::dimension::type DType> class dimension;

template <> class dimension<specfem::dimension::type::dim2> {
public:
  static constexpr auto value = specfem::dimension::type::dim2;
  static constexpr int dim = 2;
  static std::string to_string() { return "2D"; }
};

template <> class dimension<specfem::dimension::type::dim3> {
public:
  static constexpr auto value = specfem::dimension::type::dim3;
  static constexpr int dim = 3;
  static std::string to_string() { return "3D"; }
};

} // namespace dimension
} // namespace specfem

#endif /* _ENUMERATIONS_DIMENSION_HPP_ */
