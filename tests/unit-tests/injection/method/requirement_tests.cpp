// The requirements PointwiseSurfaceMethod places on a method: each broken
// method fails exactly the concept for its mistake.

#include "broken_methods.hpp"
#include "constant_force.hpp"
#include "specfem/injection/method/requirements.hpp"
#include <gtest/gtest.h>

namespace specfem::injection_method_test {

using specfem::element::medium_tag;
using specfem::injection_impl::ComputeHasQuantities;
using specfem::injection_impl::ComputeReturnsAcceleration;
using specfem::injection_impl::ComputesSomewhere;
using specfem::injection_impl::DeviceParameters;
using specfem::injection_impl::NamedMethod;
using specfem::injection_impl::PointwiseInjectionMethod;
using specfem::injection_impl::QuantitiesHaveCompute;
using specfem::injection_impl::SupportedQuantities;
using specfem::injection_impl::UniqueQuantities;

static_assert(PointwiseInjectionMethod<ConstantForce>);

static_assert(!NamedMethod<NoName>);
static_assert(ComputesSomewhere<NoName>);

static_assert(!ComputesSomewhere<NonStaticCompute>);
static_assert(!ComputesSomewhere<MutableRefCompute>);

static_assert(
    !QuantitiesHaveCompute<QuantitiesWithoutCompute, medium_tag::acoustic>);
static_assert(
    QuantitiesHaveCompute<QuantitiesWithoutCompute, medium_tag::elastic>);

static_assert(!ComputeHasQuantities<ComputeWithoutQuantities>);
static_assert(ComputesSomewhere<ComputeWithoutQuantities>);

static_assert(!UniqueQuantities<DuplicateQuantities, medium_tag::elastic>);
static_assert(SupportedQuantities<DuplicateQuantities, medium_tag::elastic>);

static_assert(!SupportedQuantities<UnsupportedQuantity, medium_tag::elastic>);
static_assert(UniqueQuantities<UnsupportedQuantity, medium_tag::elastic>);

static_assert(!ComputeReturnsAcceleration<WrongReturn>);
static_assert(ComputesSomewhere<WrongReturn>);

static_assert(!DeviceParameters<NonTrivialParameters>);
static_assert(!DeviceParameters<NoDefaultParameters>);

TEST(InjectionRequirements, ConceptsCompile) { SUCCEED(); }

} // namespace specfem::injection_method_test
