// Compile-failure cases for the requirement checks of PointwiseSurfaceMethod.
// Each target of requirement_failures.cmake defines one CASE_ macro; ctest
// builds it and matches the unsatisfied concept the compiler names.

#include "broken_methods.hpp"
#include "specfem/injection/method/pointwise_surface_method.hpp"
#if defined(CASE_VALID)
#include "constant_force.hpp"
#endif

namespace specfem::injection_method_test {

void construct_method() {
#if defined(CASE_VALID)
  [[maybe_unused]] const specfem::injection::PointwiseSurfaceMethod<
      ConstantForce> method;
#elif defined(CASE_NO_NAME)
  [[maybe_unused]] const specfem::injection::PointwiseSurfaceMethod<NoName>
      method;
#elif defined(CASE_NO_COMPUTE)
  [[maybe_unused]] const specfem::injection::PointwiseSurfaceMethod<
      NonStaticCompute> method;
#elif defined(CASE_QUANTITIES_WITHOUT_COMPUTE)
  [[maybe_unused]] const specfem::injection::PointwiseSurfaceMethod<
      QuantitiesWithoutCompute> method;
#elif defined(CASE_WRONG_RETURN)
  [[maybe_unused]] const specfem::injection::PointwiseSurfaceMethod<WrongReturn>
      method;
#else
#error "requirement_failures.cpp: define one CASE_ macro"
#endif
}

} // namespace specfem::injection_method_test
