#ifndef SPECFEM_ENUM_HPP
#define SPECFEM_ENUM_HPP

namespace specfem {
namespace enums {
namespace element {

namespace dimension {
class dim2 {
  constexpr static int dim = 2;
};
class dim3 {
  constexpr static int dim = 3;
};
} // namespace dimension

namespace quadrature {
template <int N> class quadrature_points { constexpr static int NGLL = N; };
} // namespace quadrature

namespace medium {
class elastic {};
} // namespace medium

namespace property {
class isotropic {};
} // namespace property
} // namespace element
} // namespace enums
} // namespace specfem

#endif
